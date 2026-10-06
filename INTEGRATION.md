# INTEGRATION.md — consuming decider-serve from the outside (handoff)

Written for the Godot agent (game repo `E:\godot\projects\swmg`) and any other
external client. The wrapper repo is `E:\GitRepos\decider-serve` locally,
`github.com/robbe1912/decider-serve` publicly; its `AGENTS.md` is the
runbook if you ever need to install or verify it from scratch.

## What you get

One-pass **typed decisions**: you send a state text plus questions with
option lists, you get back a calibrated probability distribution per
question. No text generation, no prompts to tune, deterministic per request.
Two models:

| model | device | latency | use |
|---|---|---|---|
| `decider-2b` | GPU (default service) | ~80 ms / request | dev-workflow router, live tooling |
| `decider-4b` (rev v2) | CPU | ~2.2–2.6 s / request | batch triage (test failures, hard items) |

The 2B service (~4.5 GiB VRAM) coexists with a running Godot editor/game on
a 10 GB card. The 4B is CPU-only by default; loading it on GPU is refused
while anything else holds the card.

## Start the service

```bat
cd /d E:\GitRepos\decider-serve
serve.bat                     :: 2B on GPU, http://127.0.0.1:8000
serve.bat 4b                  :: 4B triage model on CPU
serve.bat --port 8123         :: custom port
```

Startup takes ~16–35 s (graph capture); poll `GET /health` until
`{"ok": true, ...}`. Stop with Ctrl+C.

## The one endpoint you need: `POST /decide`

```json
{
  "context": "My card was charged twice for the same purchase.",
  "schema": {
    "Which department should handle this?": {
      "type": "choice",
      "options": ["billing", "technical support", "sales"]
    }
  }
}
```

Response — one entry per question (batch as many questions as you like into
one request; each is scored independently, one round-trip):

```json
{
  "Which department should handle this?": {
    "choice": "billing",
    "confidence": 0.9095,
    "type": "choice",
    "probabilities": {"billing": 0.9095, "technical support": 0.0404, "sales": 0.0501}
  }
}
```

Three field types (`type` is optional, defaults to `choice`):

| `"type"` | request field | response |
|---|---|---|
| `"choice"` | `{"options": [2..255 strings]}` | `{"choice", "confidence", "probabilities"}` |
| `"bool"` | none | `{"noul": p_true, "type": "noul"}` — probability the statement holds |
| `"scale"` | `{"legend": ["bad", "ok", "good"]}` or `{"0": "none", ...}` | `{"score": weighted mean, "confidence", "legend", "probabilities"}` |

Limits: 1024 scoring rows/request, ~32k tokens of state (HTTP 413 above
that; 422 on malformed schema; 503 when the queue is full — retry).
`GET /health`, `GET /v1/models`, `GET /stats` for ops. `POST /v1/systemone`
exists for the structured Jev format; `/decide` is the simple one.

## Godot 4 client (GDScript)

Dev-tooling client, not a shipped gameplay dependency. Add as a node, wire
the signal, reuse one instance (HTTPRequest per concurrent call).

```gdscript
class_name DeciderClient
extends Node
## HTTP client for the local decider-serve wrapper. Dev tooling only.

signal decision_received(result: Dictionary)
signal request_failed(reason: String)

const DEFAULT_URL: String = "http://127.0.0.1:8000"
const TIMEOUT_SEC: float = 30.0

@export var base_url: String = DEFAULT_URL

var _http: HTTPRequest

func _ready() -> void:
	if Engine.is_editor_hint():
		return
	_http = HTTPRequest.new()
	_http.timeout = TIMEOUT_SEC
	_http.request_completed.connect(_on_request_completed)
	add_child(_http)

## state: context text; schema: {question: {"type": "choice", "options": [...]}, ...}
func decide(state: String, schema: Dictionary) -> void:
	if Engine.is_editor_hint():
		return
	var body := JSON.stringify({"context": state, "schema": schema})
	var headers := PackedStringArray(["Content-Type: application/json"])
	var err := _http.request(base_url + "/decide", headers, HTTPClient.METHOD_POST, body)
	if err != OK:
		request_failed.emit("request failed to start (error %d)" % err)

func _on_request_completed(_result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if code != 200:
		request_failed.emit("HTTP %d: %s" % [code, body.get_string_from_utf8()])
		return
	var json := JSON.new()
	if json.parse(body.get_string_from_utf8()) != OK:
		request_failed.emit("invalid JSON from server")
		return
	decision_received.emit(json.data)

## Helper: highest-probability option of a choice answer.
static func best_option(answer: Dictionary) -> StringName:
	var probs: Dictionary = answer.get("probabilities", {})
	var best: StringName = &""
	var best_p := -1.0
	for key: String in probs:
		if float(probs[key]) > best_p:
			best_p = float(probs[key])
			best = StringName(key)
	return best
```

Usage (router example — one request, three questions):

```gdscript
@onready var decider: DeciderClient = $DeciderClient

func _ready() -> void:
	decider.decision_received.connect(_on_decision)
	decider.decide(
		"Test run failed after 42 of 260 tests. Godot 4.7, renderer forward_plus. Last logs: physics step took 340ms, then assertion in save_game().",
		{
			"Which subsystem is the likely culprit?":
				{"type": "choice", "options": ["physics", "save system", "rendering", "networking"]},
			"Is this a regression rather than a flaky test?":
				{"type": "bool"},
			"How severe is the failure?":
				{"type": "scale", "legend": ["minor", "moderate", "blocking"]}
		})

func _on_decision(result: Dictionary) -> void:
	print(DeciderClient.best_option(result["Which subsystem is the likely culprit?"]))
	print(result["Is this a regression rather than a flaky test?"]["noul"])
	print(result["How severe is the failure?"]["score"])
```

## CLI for scripts / CI (journals every decision)

```bat
cd /d E:\GitRepos\decider-serve
decide.bat --url http://127.0.0.1:8000 --state-file failure.log ^
  -q "Which subsystem caused this test failure?" --opts "physics,save system,rendering,networking"
```

Prints the JSON answer on stdout and appends `ts, model, question, options,
probs, chosen` to `journal.jsonl` — the calibration feedback loop for later.
Without `--url` it loads the model in-process (slower start, same result).

## Rules for consumers

1. Talk HTTP to the wrapper; never load the models inside the game process.
2. The wrapper folder is not yours to modify — commands above are the API.
   (The game repo is likewise untouched by the wrapper; separation is the
   point.)
3. 4B on GPU requires the game/editor closed; the service enforces it.
4. Weights never enter any git repo; they live in `decider-serve\models\`.
5. Expected behavior is documented and stable: canonical example above
   returns billing ≈ 0.91 (2B) / 0.914 (4B v2). If you see materially
   different numbers, something is wrong — check `/health`.

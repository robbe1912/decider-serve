"""CLI for one-off decisions against a local decider model (or a running server).

  python decide.py --state-file s.txt -q "Which department?" --opts billing,support,sales
  python decide.py --state "text..." -q "Q1" --opts a,b -q "Q2" --opts yes,no

Repeatable -q/--opts pairs, matched in order. Prints JSON probabilities and
appends one journal.jsonl line per question (timestamp, model, question,
options, probs, chosen) for the future calibration feedback loop.

  --model 2b|4b|path   (default $DECIDER_MODEL or 2b; ignored with --url)
  --device auto|cuda|cpu
  --url http://host:port   send the request to a running serve.py instead of
                           loading the weights in-process (keeps VRAM single-tenant)
"""
import argparse
import datetime
import json
import sys
from pathlib import Path
import compat  # torch<2.6 shim: must run before decider/transformers import

REPO = Path(__file__).resolve().parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--state-file", help="read the state text from this file")
    src.add_argument("--state", help="the state text inline")
    ap.add_argument("-q", "--question", action="append", required=True, help="question text (repeatable)")
    ap.add_argument("--opts", action="append", required=True,
                    help="comma-separated option list for the preceding -q (repeatable)")
    ap.add_argument("--model", default=None, help="2b | 4b | models/<name> | folder path")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--url", default=None, help="decide via a running server instead of in-process")
    ap.add_argument("--journal", default=str(REPO / "journal.jsonl"))
    args = ap.parse_args(argv)

    if len(args.question) != len(args.opts):
        ap.error(f"{len(args.question)} -q but {len(args.opts)} --opts: they come in pairs")
    questions = []
    for q, raw in zip(args.question, args.opts):
        opts = [o.strip() for o in raw.split(",")]
        if not (2 <= len(opts) <= 255) or any(not o for o in opts):
            ap.error(f'question "{q}": need 2..255 non-empty options, got {raw!r}')
        questions.append((q, opts))
    args.questions = questions
    return args


def http_decide(url, state, questions):
    import httpx
    schema = {q: {"type": "choice", "options": opts} for q, opts in questions}
    r = httpx.post(url.rstrip("/") + "/decide", json={"context": state, "schema": schema}, timeout=300)
    r.raise_for_status()
    data = r.json()  # {question: {"choice", "confidence", "probabilities", "type"}}
    return [{"question": q, "choice": data[q]["choice"], "confidence": data[q].get("confidence"),
             "probs": data[q]["probabilities"]} for q, _ in questions]


def local_decide(path, device, state, questions, label):
    import torch
    from common import fits_on_gpu
    from decider.infer import Decider
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            fits, free = fits_on_gpu(path)
            if not fits:
                print(f"[decide] {label} does not fit in free VRAM ({free / 1024**3:.1f} GiB free) -- using cpu",
                      file=sys.stderr)
                device = "cpu"
    d = Decider(path, device=device, use_graphs=False)  # eager: torch 2.5.1 inductor != triton-windows 3.8 API
    res = d.decide(state, [{"question": q, "options": opts} for q, opts in questions])
    out = [{"question": q, "choice": r["choice"], "confidence": r["confidence"], "probs": r["probs"]}
           for (q, _), r in zip(questions, res)]
    return device, d.name, out


def main(argv=None):
    args = parse_args(argv)
    state = Path(args.state_file).read_text(encoding="utf-8") if args.state_file else args.state

    if args.url:
        model_label = args.url
        out = http_decide(args.url, state, args.questions)
    else:
        from common import model_dir
        path, label = model_dir(args.model)
        device, model_label, out = local_decide(path, args.device, state, args.questions, label)
        print(f"[decide] model={model_label} device={device}", file=sys.stderr)

    print(json.dumps(out, indent=2))

    with open(args.journal, "a", encoding="utf-8") as j:
        for row in out:
            j.write(json.dumps({
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "model": model_label,
                "question": row["question"],
                "options": list(row["probs"]),
                "probs": row["probs"],
                "chosen": row["choice"],
            }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

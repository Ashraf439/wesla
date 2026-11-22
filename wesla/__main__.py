"""Command line entry point.

  python -m wesla ask "What did Meta release this week?"
  python -m wesla chat                 # interactive, with +/- feedback
  python -m wesla ui                   # Gradio app
  python -m wesla stats
  python -m wesla push-queue           # upload the training queue to the HF dataset repo
  python -m wesla sync runs/wesla      # apply a trainer run (columns.json + trained.json)
  python -m wesla prune                # evict decayed, unused LTM items
  python -m wesla demo                 # fully offline demo with canned web results
  python -m wesla doctor               # check every dependency (Ollama, models, search, ...)
  python -m wesla logs                 # show the end of data/logs/wesla.log

Add --log-level DEBUG before the command for verbose console output
(the log file always keeps DEBUG detail).
"""
from __future__ import annotations

import argparse
import json
import sys


def _print_answer(ans) -> None:
    print(f"\n{ans.text}\n")
    print(f"  route: {ans.route.value}  ({ans.debug.get('route_reason')})  model: {ans.debug.get('model')}")
    for s in ans.sources[:5]:
        print(f"  source: {s}")
    if ans.debug.get("decision_counts"):
        print(f"  learning judge: {ans.debug['decision_counts']}")
    print(f"  answer id: {ans.debug['answer_id']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wesla", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--log-level", default=None, help="console level: DEBUG, INFO (default), WARNING")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask"); p.add_argument("question", nargs="+")
    sub.add_parser("chat")
    p = sub.add_parser("ui"); p.add_argument("--share", action="store_true"); p.add_argument("--port", type=int, default=7860)
    sub.add_parser("stats")
    sub.add_parser("push-queue")
    p = sub.add_parser("sync"); p.add_argument("run_dir")
    sub.add_parser("prune")
    sub.add_parser("demo")
    p = sub.add_parser("doctor"); p.add_argument("--quick", action="store_true", help="skip network checks")
    p = sub.add_parser("logs"); p.add_argument("-n", type=int, default=80)
    args = ap.parse_args(argv)

    from .config import load_config
    from .log import get_logger, setup_logging

    cfg = load_config(args.config)
    log_file = setup_logging(f"{cfg.data_dir}/logs", args.log_level or ("WARNING" if args.cmd == "demo" else None))
    log = get_logger("cli")
    log.info("command: %s", " ".join(sys.argv[1:]) or args.cmd)

    if args.cmd == "logs":
        if log_file.exists():
            print("\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-args.n:]))
        crash = log_file.parent / "crash.log"
        if crash.exists() and "Fatal Python error" in crash.read_text(encoding="utf-8", errors="replace"):
            print(f"\n*** native crash traces found in {crash} ***")
        return 0
    if args.cmd == "doctor":
        from .doctor import run_doctor

        return run_doctor(cfg, quick=args.quick)

    if args.cmd == "demo":
        from .demo import run_demo

        run_demo()
        return 0
    if args.cmd == "ui":
        from .app import launch

        launch(args.config, share=args.share, port=args.port, log_level=args.log_level)
        return 0

    from .agent import WESLA

    agent = WESLA(cfg)
    if args.cmd == "ask":
        try:
            _print_answer(agent.ask(" ".join(args.question)))
        except Exception as exc:
            log.exception("ask failed")
            print(f"\nERROR: {type(exc).__name__}: {exc}\nFull traceback: {log_file}")
            return 1
    elif args.cmd == "chat":
        print("WESLA chat. Commands: '+' / '-' rate last answer, '-! <correction>' to correct, 'quit'.")
        last = None
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in {"quit", "exit"}:
                break
            if q.startswith(("+", "-")) and last:
                corr = q[2:].strip() if q.startswith("-!") else ""
                print(agent.feedback(last, 1 if q.startswith("+") else -1, corr))
                continue
            if q:
                try:
                    ans = agent.ask(q)
                except Exception as exc:
                    log.exception("ask failed")
                    print(f"ERROR: {type(exc).__name__}: {exc}  (traceback in {log_file})")
                    continue
                _print_answer(ans)
                last = ans.debug["answer_id"]
    elif args.cmd == "stats":
        print(json.dumps(agent.stats(), indent=2))
    elif args.cmd == "push-queue":
        print(agent.queue.push_to_hub())
    elif args.cmd == "sync":
        print(agent.sync_training_results(args.run_dir))
    elif args.cmd == "prune":
        print(f"pruned {agent.ltm.prune()} item(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Logging for WESLA.

setup_logging() sends logs to the console and to a rotating file
(data/logs/wesla.log by default). It also:
  * enables `faulthandler`, so native crashes (segfaults, the Windows OpenMP
    abort from faiss + torch, ...) still leave a stack trace in
    data/logs/crash.log even though Python itself dies;
  * installs hooks so uncaught exceptions in the main thread and in worker
    threads (Gradio runs handlers in threads) are written to the log.

Use `stage()` to time a block:
    with stage(log, "web search", query=q):
        ...
It logs "start", "done in X.XXs" or the full traceback on failure.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_CONFIGURED = False
_CRASH_FH = None  # keep the crash-log file handle open for faulthandler

FMT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("wesla") else f"wesla.{name}")


def setup_logging(log_dir: str = "data/logs", level: str | None = None) -> Path:
    """Idempotent. Returns the log file path. Level from arg, else WESLA_LOG_LEVEL, else INFO."""
    global _CONFIGURED, _CRASH_FH
    log_path = Path(log_dir) / "wesla.log"
    if _CONFIGURED:
        return log_path
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    lvl = getattr(logging, (level or os.environ.get("WESLA_LOG_LEVEL", "INFO")).upper(), logging.INFO)

    root = logging.getLogger("wesla")
    root.setLevel(logging.DEBUG)          # handlers filter
    root.propagate = False

    fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)            # the file always keeps full detail
    fh.setFormatter(logging.Formatter(FMT))
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(lvl)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(fh)
    root.addHandler(ch)

    # quieten chatty libraries, but keep their warnings
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "huggingface_hub",
                  "trafilatura", "primp", "gradio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # native crashes -> crash.log (append, with a header per process start)
    _CRASH_FH = open(Path(log_dir) / "crash.log", "a", encoding="utf-8")
    _CRASH_FH.write(f"\n===== process {os.getpid()} started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    _CRASH_FH.flush()
    faulthandler.enable(file=_CRASH_FH, all_threads=True)

    log = logging.getLogger("wesla")

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def _thread_hook(args):
        log.critical("Uncaught exception in thread %s", args.thread.name if args.thread else "?",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook
    _CONFIGURED = True
    log.info("logging to %s (console level %s); native crashes -> %s",
             log_path.resolve(), logging.getLevelName(lvl), (Path(log_dir) / "crash.log").resolve())
    return log_path


@contextmanager
def stage(log: logging.Logger, name: str, **ctx):
    from .activity import checkpoint
    labels = {"select column":"Selecting a model", "route":"Checking memory and model confidence",
              "web search":"Searching the web", "fetch + chunk pages":"Reading source pages",
              "verify":"Verifying sources", "learning judge":"Evaluating evidence", "generate answer":"Writing the answer"}
    checkpoint(labels.get(name))
    extra = " ".join(f"{k}={str(v)[:80]!r}" for k, v in ctx.items())
    log.info(">> %s %s", name, extra)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception("!! %s FAILED after %.2fs", name, time.perf_counter() - t0)
        raise
    else:
        log.info("<< %s done in %.2fs", name, time.perf_counter() - t0)

"""Gradio UI: chat + routing/verification/judge inspector + feedback + stats.

Every handler is wrapped: an exception is written to data/logs/wesla.log with a
full traceback and shown in the chat, so the server keeps running.
"""
from __future__ import annotations

import json
import traceback

from .log import get_logger, setup_logging

log = get_logger("app")


def _make_chatbot(gr):
    try:
        return gr.Chatbot(type="messages", height=480)
    except TypeError:  # Gradio >= 6: messages format is the default and `type` was removed
        return gr.Chatbot(height=480)


def launch(config_path: str | None = None, share: bool = False, port: int = 7860,
           log_level: str | None = None) -> None:
    from .config import load_config

    cfg = load_config(config_path)
    log_file = setup_logging(f"{cfg.data_dir}/logs", log_level)

    import gradio as gr

    from .agent import WESLA

    log.info("gradio %s; starting agent", getattr(gr, "__version__", "?"))
    agent = WESLA(cfg)

    def respond(message, history, state):
        history = list(history or [])
        history.append({"role": "user", "content": message})
        try:
            ans = agent.ask(message)
        except Exception as exc:
            log.exception("query failed: %r", message[:200])
            err = (f"**Error:** {type(exc).__name__}: {exc}\n\n"
                   f"Full traceback in `{log_file}`.")
            history.append({"role": "assistant", "content": err})
            return history, "", state, {"error": str(exc), "traceback": traceback.format_exc()[-2000:]}, [], []
        state = ans.debug["answer_id"]
        srcs = "\n".join(f"- {s}" for s in ans.sources[:6])
        reply = ans.text + (f"\n\n**Sources**\n{srcs}" if srcs else "")
        history.append({"role": "assistant", "content": reply})
        decisions = [{"action": d.action.value, "L_s": round(d.score, 3),
                      **{k: round(v, 2) for k, v in d.components.items()}, "why": d.rationale}
                     for d in ans.decisions]
        info = {"route": ans.route.value, **{k: v for k, v in ans.debug.items() if k != "verification"}}
        return history, "", state, info, ans.debug.get("verification", []), decisions

    def rate(state, good, correction):
        if not state:
            return "Ask something first."
        try:
            r = agent.feedback(state, 1 if good else -1, correction or "")
        except Exception as exc:
            log.exception("feedback failed")
            return f"Feedback failed: {exc}"
        return f"Feedback recorded; updated memory items {r['updated_memory_ids']}"

    def safe(fn):
        def wrapped(*a):
            try:
                return fn(*a)
            except Exception as exc:
                log.exception("%s failed", getattr(fn, "__name__", "action"))
                return f"Failed: {exc}"
        return wrapped

    def tail_log(n: int = 200) -> str:
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except FileNotFoundError:
            return "(no log yet)"

    with gr.Blocks(title="WESLA") as demo:
        gr.Markdown("## WESLA — Web-Enhanced Self-Learning Agent")
        state = gr.State("")
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chat = _make_chatbot(gr)
                    box = gr.Textbox(placeholder="Ask anything…", show_label=False)
                    with gr.Row():
                        up = gr.Button("👍 Helpful")
                        down = gr.Button("👎 Wrong")
                    corr = gr.Textbox(label="Correction (optional, sent with 👎)")
                    fb = gr.Markdown()
                with gr.Column(scale=2):
                    info = gr.JSON(label="Router / model")
                    ver = gr.JSON(label="Verification layer")
                    dec = gr.JSON(label="Learning Judge decisions")
            box.submit(respond, [box, chat, state], [chat, box, state, info, ver, dec])
            up.click(lambda s: rate(s, True, ""), [state], [fb])
            down.click(lambda s, c: rate(s, False, c), [state, corr], [fb])
        with gr.Tab("Memory & learning"):
            stats = gr.JSON()
            gr.Button("Refresh").click(lambda: agent.stats(), None, stats)
            run_dir = gr.Textbox(label="Trainer run directory to sync (columns.json + trained.json)")
            out = gr.Markdown()
            gr.Button("Sync training results").click(
                safe(lambda p: json.dumps(agent.sync_training_results(p))), [run_dir], [out])
            gr.Button("Push queue to HF Hub").click(safe(lambda: agent.queue.push_to_hub()), None, out)
        with gr.Tab("Logs"):
            gr.Markdown(f"Log file: `{log_file}`")
            logbox = gr.Textbox(value=tail_log, lines=30, max_lines=30, show_label=False)
            gr.Button("Refresh log").click(lambda: tail_log(), None, logbox)
    log.info("UI at http://127.0.0.1:%d", port)
    if share:
        demo.launch(share=True, server_port=port, show_error=True)
    else:
        from .workspace_ui import serve
        serve(agent, demo, port)


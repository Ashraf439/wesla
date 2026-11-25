"""Regression tests: router confidence calibration and best-effort OCR."""
import subprocess
import sys

import pytest

from wesla import visual
from wesla.config import Config, load_config
from wesla.types import Route
from tests.test_agent_and_trainer import RESULTS, make_agent


# ------------------------------------------------------------------ router thresholds
@pytest.mark.parametrize("cfg", [Config(), load_config("config.toml")], ids=["defaults", "config.toml"])
def test_router_thresholds_are_calibrated(cfg):
    # llama3.2:1b gave wrong answers at 0.56-0.92 nats/token; off-topic memory reached 0.72
    assert cfg.router.entropy_threshold <= 0.5
    assert 0.72 < cfg.memory.hit_threshold <= 0.75


def test_moderately_uncertain_draft_goes_to_web(tmp_path):
    # entropy 0.92 is what "LoRA = Licensed Assisted Mobility" scored; it must not be trusted
    a = make_agent(tmp_path, RESULTS, entropy=0.92)
    ans = a.ask("define a neural network")
    assert ans.route == Route.WEB
    assert "entropy 0.92" in ans.debug["route_reason"]


# ------------------------------------------------------------------ OCR
def test_ocr_timeout_returns_empty_text(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="powershell.exe", timeout=visual.OCR_PAGE_TIMEOUT_S)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", slow)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    assert visual.read_text(b"png") == ""


def test_ocr_engine_failure_returns_empty_text(monkeypatch):
    def broken():
        raise ImportError("rapidocr_onnxruntime is not installed")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(visual, "ocr_engine", broken)
    assert visual.read_text(b"png") == ""


def test_scanned_pdf_ocr_stops_at_time_budget(monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for _ in range(5):
        doc.new_page()          # blank pages look scanned (no text), so each wants OCR
    data = doc.tobytes()

    calls = []
    clock = iter([0, 0, 50, 100, 150, 200, 250, 300])   # start, then one tick per page
    monkeypatch.setattr(visual, "monotonic", lambda: next(clock))
    monkeypatch.setattr(visual, "read_text", lambda png: calls.append(1) or "")
    pages = list(visual.scan_aware_pdf(data))

    assert len(pages) == 5                     # every page is still returned
    assert len(calls) == 2                     # but OCR stopped once 90 s were used

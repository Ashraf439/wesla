"""Per-request progress and cooperative cancellation."""
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event

class Stopped(Exception):
    pass

@dataclass
class Activity:
    label: str = "Waiting for the local model"
    cancelled: Event = field(default_factory=Event)

current = ContextVar("wesla_activity", default=None)

def checkpoint(label=None):
    job = current.get()
    if job:
        if job.cancelled.is_set():
            raise Stopped("Stopped")
        if label:
            job.label = label


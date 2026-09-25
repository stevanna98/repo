"""Bound test CPU threads; allow headless figure generation in project output."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"))
import pytest
import torch
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True, scope="session")
def bounded_threads():
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        yield


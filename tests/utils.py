from __future__ import annotations

import json
import sys
from pathlib import Path

SAMPLE_BUNDLE = Path(__file__).resolve().parents[1] / "examples" / "sample_run_bundle.json"


def load_sample_bundle() -> dict:
    return json.loads(SAMPLE_BUNDLE.read_text())


def reset_backend_modules() -> None:
    for name in ["gce.core.cc_surface.api", "cc.core.api", "cc.core", "cc"]:
        sys.modules.pop(name, None)

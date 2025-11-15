from __future__ import annotations

import importlib
from pathlib import Path

from gce.exporters import one_page

from tests.utils import load_sample_bundle, reset_backend_modules


def _compute_sample_verdict():
    reset_backend_modules()
    import gce.core.cc_surface.api as api

    importlib.reload(api)
    bundle = load_sample_bundle()
    return bundle, api.compute_verdict_from_params(**bundle)


def test_render_contains_sections():
    bundle, verdict = _compute_sample_verdict()
    report = one_page.render_text_report(bundle, verdict)
    assert "Guardrail One-Pager" in report
    assert bundle["rule"] in report
    assert verdict.label in report
    assert "Next Tests" in report
    assert "Checklist" in report


def test_export_roundtrip(tmp_path: Path):
    bundle, verdict = _compute_sample_verdict()
    target = tmp_path / "bundle.txt"
    path = one_page.export_one_pager(bundle, verdict, target)
    assert path == target
    assert target.read_text() == one_page.render_text_report(bundle, verdict)


def test_build_payload_is_json_ready():
    bundle, verdict = _compute_sample_verdict()
    payload = one_page.build_payload(bundle, verdict)
    assert payload["bundle"]["rule"] == bundle["rule"]
    assert payload["verdict"]["label"] == verdict.label

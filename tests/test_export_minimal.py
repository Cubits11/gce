from __future__ import annotations

import importlib
import json
from pathlib import Path

from gce.exporters import one_pager
from tests.utils import load_sample_bundle, reset_backend_modules


def _compute_sample_verdict():
    """
    Reload the CC-surface API using a clean module state and compute a sample verdict.

    This ensures:
    - The backend selection logic (fallback vs cc-framework) is exercised.
    - We always use the same canonical sample bundle for all exporter tests.
    """
    reset_backend_modules()
    import gce.core.cc_surface.api as api  # local import to respect reset

    importlib.reload(api)
    bundle = load_sample_bundle()
    verdict = api.compute_verdict_from_params(**bundle)
    return bundle, verdict


def test_render_contains_sections():
    """
    Basic smoke test: the text one-pager should contain key sections and content.

    We don't assert exact wording (that would make copy edits too brittle), but we
    do require that:
    - The default title appears.
    - The bundle rule name appears.
    - The verdict label appears.
    - Both "Next Tests" and "Checklist" section headers are present.
    """
    bundle, verdict = _compute_sample_verdict()
    report = one_pager.render_text_report(bundle, verdict)

    assert "Guardrail One-Pager" in report
    assert bundle["rule"] in report
    assert verdict.label in report
    assert "Next Tests" in report
    assert "Checklist" in report

    # Basic sanity: report ends with a newline (nice for CLI tools / POSIX streams).
    assert report.endswith("\n")


def test_export_roundtrip(tmp_path: Path):
    """
    Export a text one-pager to disk and ensure the contents match render_text_report.

    This guarantees that:
    - export_one_pager respects the target path.
    - The written file is a pure text representation of render_text_report.
    """
    bundle, verdict = _compute_sample_verdict()
    expected = one_pager.render_text_report(bundle, verdict)

    target = tmp_path / "bundle.txt"
    path = one_pager.export_one_pager(bundle, verdict, target)

    assert path == target
    assert target.exists()
    assert target.read_text() == expected


def test_build_payload_is_json_ready():
    """
    build_payload should produce a fully JSON-serialisable dict.

    We check:
    - bundle + verdict are embedded under the expected keys.
    - json.dumps succeeds without needing custom encoders.
    """
    bundle, verdict = _compute_sample_verdict()
    payload = one_pager.build_payload(bundle, verdict)

    # Structural expectations
    assert payload["bundle"]["rule"] == bundle["rule"]
    assert payload["verdict"]["label"] == verdict.label

    # Should be fully JSON-serialisable as-is
    json_blob = json.dumps(payload)
    assert isinstance(json_blob, str)
    assert '"bundle"' in json_blob
    assert '"verdict"' in json_blob


def test_verdict_to_json_embeds_bundle_and_metadata():
    """
    verdict_to_json should include verdict, optional bundle, and optional metadata.

    We sanity-check that:
    - The JSON parses back to a dict.
    - The metadata keys are present when provided.
    """
    bundle, verdict = _compute_sample_verdict()
    meta = {"source": "pytest", "kind": "unit-test"}

    json_blob = one_pager.verdict_to_json(
        verdict,
        bundle=bundle,
        metadata=meta,
    )
    parsed = json.loads(json_blob)

    assert parsed["bundle"]["rule"] == bundle["rule"]
    assert parsed["verdict"]["label"] == verdict.label
    assert parsed["metadata"] == meta
    assert "generated_at" in parsed


def test_verdict_to_pdf_creates_nonempty_pdf(tmp_path: Path):
    """
    verdict_to_pdf should create a single PDF file with non-zero size.

    We don't inspect the PDF contents in detail (that's a job for reportlab),
    but we do assert:
    - The file exists at the requested path.
    - It has a .pdf suffix.
    - It has non-zero length.
    """
    _, verdict = _compute_sample_verdict()
    target = tmp_path / "verdict.pdf"

    path = one_pager.verdict_to_pdf(verdict, output_path=target)
    assert path == target
    assert path.exists()
    assert path.suffix.lower() == ".pdf"
    assert path.stat().st_size > 0

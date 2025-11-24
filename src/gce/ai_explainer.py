from __future__ import annotations

"""
Lightweight AI helper for GCE.

- Primary path: call an OpenAI chat model to explain a RunBundle + Verdict
  in plain English for humans.
- Fallback path: if the OpenAI client or API key are missing, return a
  deterministic, non-LLM summary so the UI still works offline.

This keeps tests and offline usage safe while giving you a real AI feature
when OPENAI_API_KEY is configured.
"""

import json
import os
from typing import Optional

from openai import OpenAI  # type: ignore[import-untyped]

from .core.cc_surface.api import RunBundle, Verdict

# Default model; can be overridden via env var GCE_AI_MODEL.
_DEFAULT_MODEL = os.getenv("GCE_AI_MODEL", "gpt-4.1-mini")

_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """
    Lazily construct an OpenAI client if an API key is available.

    Returns None if:
    - the openai package is not usable, or
    - OPENAI_API_KEY is not set, or
    - client construction fails for any reason.

    This ensures importing this module never crashes tests.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No key = operate in offline/fallback mode.
        return None

    try:
        _client = OpenAI()
    except Exception:
        # Fail closed: treat as unavailable rather than crashing.
        _client = None
    return _client


def _fallback_explanation(bundle: RunBundle, verdict: Verdict) -> str:
    """
    Deterministic explanation used when no LLM is available.

    This still gives the user a readable story and makes it obvious that
    the AI helper is not configured, which is useful both for demos and tests.
    """
    num_baselines = len(bundle.J_baselines)
    baseline_names = ", ".join(sorted(bundle.J_baselines.keys())) or "none"

    return (
        "AI explanation is running in **offline mode** because no OpenAI API key "
        "is configured.\n\n"
        f"- Composition rule: `{bundle.rule}` at θ={bundle.theta:.2f}\n"
        f"- Objective: `{bundle.objective}` over {num_baselines} singleton(s): {baseline_names}\n"
        f"- Composed J: {bundle.J_composed:.3g}\n"
        f"- Composability verdict: **{verdict.label}** with CC={verdict.CC:.2f}\n\n"
        "Interpretation:\n"
        f"- A **{verdict.label}** label means the composed guardrails behave "
        "roughly as classified by the CC metric.\n"
        "- To get a richer AI narrative, set the `OPENAI_API_KEY` environment "
        "variable and re-run the tool."
    )


def _build_prompt(bundle: RunBundle, verdict: Verdict) -> str:
    """
    Build a compact JSON-based prompt for the LLM.

    We keep it structured so it's easy to debug and reason about.
    """
    bundle_data = bundle.model_dump()
    verdict_data = verdict.model_dump()

    return (
        "You are an expert safety-engineering coach for AI guardrail composition.\n"
        "Given a run bundle (experiment settings) and a verdict (label + CC metric), "
        "explain in clear language what this result means and what the team should "
        "do next.\n\n"
        "Requirements:\n"
        "- Start with a short 2–3 sentence summary.\n"
        "- Then provide 3–5 bullet points with concrete insights.\n"
        "- End with 1–2 suggested next tests.\n"
        "- Keep it under 250 words.\n\n"
        f"RunBundle JSON:\n{json.dumps(bundle_data, indent=2)}\n\n"
        f"Verdict JSON:\n{json.dumps(verdict_data, indent=2)}\n"
    )


def explain_with_ai(bundle: RunBundle, verdict: Verdict) -> str:
    """
    Main entry point used by the Gradio UI.

    - If an OpenAI client is available, call a chat model and return Markdown.
    - If anything fails (no key, network error, etc.), return a safe fallback.
    """
    client = _get_client()
    if client is None:
        return _fallback_explanation(bundle, verdict)

    prompt = _build_prompt(bundle, verdict)

    try:
        completion = client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise, technically-accurate guardrail "
                        "composability coach. Respond in Markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = completion.choices[0].message.content or ""
        return content.strip()
    except Exception as exc:  # pragma: no cover - defensive
        # Never let API issues crash the app; fall back with context.
        base = _fallback_explanation(bundle, verdict)
        return f"{base}\n\n_AI call failed; falling back to offline explanation._\n\nDetails: `{exc}`"

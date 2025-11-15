# Guardrail Composability Explorer (GCE)

Measure whether combining AI guardrails is **Constructive**, **Independent**, or **Destructive**.
Paste a run bundle → get CC verdict, one-line recommendation, and next tests in under 2 minutes.

## Quickstart
```bash
uv venv && uv pip install -e .
python -m src.gce.app.gradio_app

Open the printed URL. Paste examples/sample_run_bundle.json. Click Compute:

Label chip (🟢/🟡/🔴)

One-line recommendation

3–4 Next Tests

Export JSON / Download 1-pager

Method (short)

J = TPR − FPR (Youden’s J)

CC = J_comp / max(J_baselines)

Thresholds:

CC < 0.95: Constructive

0.95 ≤ CC ≤ 1.05: Independent

CC > 1.05: Destructive

Screenshots

(…add 2–3 images…)

Limitations & Ethics

Toy Bernoulli model; synthetic data; not a production security tool.

License

Code: MIT (or Apache-2.0). Docs/images: CC-BY-4.0.

# GCE — Guardrail Composability Explorer

GCE is the operator-facing explorer for the CC-Framework.

It turns compositional AI safety theory into an interactive surface where researchers, engineers, and auditors can inspect how guardrail combinations behave compared to their individual baselines.

## Role in the Stack

GCE is not the mathematical kernel. That is cc-framework.

GCE is not the enterprise assurance platform. That is SysESoft AssuranceKit.

GCE is the thin, usable interface between the two.

It provides:

- a stable API facade over CC computation
- local fallback behavior when cc-framework is not installed
- composability coefficient calculation
- constructive / independent / destructive classification
- CLI utilities for quick inspection
- a future UI surface for demonstrations, reports, and teaching

## Why This Exists

Most AI safety evaluations ask whether a single guardrail performs well.

GCE asks a harder question:

What happens when guardrails are composed?

A system can look safe at the individual level and become fragile at the composition level. GCE exists to make that failure visible before deployment.

## Quick Start

```bash
git clone https://github.com/Cubits11/gce.git
cd gce
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest

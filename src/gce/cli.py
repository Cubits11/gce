from __future__ import annotations

"""
GCE CLI utilities.

Two main entry points (also exposed as console scripts via pyproject.toml):

- `gce-backend-info`  → print_backend_info()
- `gce-cc-quickcheck` → cc_quickcheck()

You can also run:

    python -m gce.cli

to use the Typer-powered multi-command interface.
"""

from typing import Any

import typer

from gce.core.cc_surface.api import backend_info, compute_verdict_from_params

app = typer.Typer(help="Guardrail Composability Explorer (GCE) CLI utilities.")


@app.command("backend-info")
def backend_info_cli() -> None:
    """
    Typer command: print information about the active backend.

    This is a thin wrapper over `print_backend_info`, which is also used as the
    console-script entry point (`gce-backend-info`).
    """
    print_backend_info()


def print_backend_info() -> None:
    """
    Console-script entry for `gce-backend-info`.

    Prints a small key → value listing of the active backend configuration.
    """
    info = backend_info()
    for key, value in info.items():
        print(f"{key}: {value}")


@app.command("cc-quickcheck")
def cc_quickcheck_cli() -> None:
    """
    Typer command: run a tiny CC smoke test and print the verdict as JSON.

    This mirrors the `cc_quickcheck` console-script entry point (`gce-cc-quickcheck`)
    but is wired into the Typer app so it can be invoked via:

        python -m gce.cli cc-quickcheck
    """
    cc_quickcheck()


def cc_quickcheck() -> None:
    """
    Console-script entry for `gce-cc-quickcheck`.

    Runs a tiny, hard-coded smoke check:

    - Two singleton baselines (A, B)
    - One composed configuration
    - objective = "minimize" (smaller J is better)

    and prints the resulting Verdict as pretty-printed JSON.
    """
    verdict = compute_verdict_from_params(
        theta=0.5,
        patterns=["demo"],
        rule="SEQUENTIAL(DFA→RR)",
        J_baselines={"A": 0.30, "B": 0.40},
        J_composed=0.28,
        objective="minimize",
    )

    # Pydantic v2: model_dump_json returns a JSON string
    json_blob: str = verdict.model_dump_json(indent=2)  # type: ignore[assignment]
    print(json_blob)


if __name__ == "__main__":
    # When invoked as `python -m gce.cli`, run the Typer multi-command app.
    app()

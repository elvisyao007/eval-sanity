"""eval-sanity v0.3 demo — deterministic agent trajectory evaluation.

Runnable with plain ``python examples/trajectory_demo.py``. No models, no
network. Optional: pass ``--traces-dir PATH`` to load real Phase A traces from
an invoice-agent run.

We synthesize four scenarios:

  A. GOOD trace — correct 3-step flow: extract → validate(pass) → write_back.
     Evaluator should PASS.

  B. BAD: skip validate — extract → write_back with no validate step.
     Evaluator catches the order violation.

  C. BAD: failed validate then write_back anyway — buggy agent that ignores the
     validate result and writes regardless.
     Evaluator catches the constraint on 'passed' field.

  D. BAD: redundant + wrong tool — redundant extract call AND an unexpected
     'summarize_invoice' call inserted mid-trace.
     Evaluator catches both.

  E. REGRESSION — baseline set of good traces vs candidate set where half the
     agents skip validate. detect_trajectory_regression flags the drop.

If ``--traces-dir`` is given, the demo also loads and evaluates the real invoice-
agent traces to show dogfooding of the Phase A outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from eval_sanity import (
    OrderConstraint,
    TrajectorySpec,
    detect_trajectory_regression,
    trajectory_report,
)
from eval_sanity.trajectory import Trajectory, TrajectoryStep


# ---------------------------------------------------------------------------
# Specs for the invoice-agent task
# ---------------------------------------------------------------------------

SUCCESS_SPEC = TrajectorySpec(
    expected_tools=["extract_fields", "validate", "write_back"],
    forbidden_tools=[],
    order_constraints=[
        OrderConstraint(
            tool="write_back",
            required_predecessor="validate",
            require_pass_field="passed",
        )
    ],
    max_steps=5,
    expected_final_status="success",
)

FLAGGED_SPEC = TrajectorySpec(
    expected_tools=["extract_fields", "validate"],
    forbidden_tools=["write_back"],
    order_constraints=[],
    max_steps=4,
    expected_final_status="flagged",
)


# ---------------------------------------------------------------------------
# Synthetic trace builders
# ---------------------------------------------------------------------------


def _step(n, tool, args, result, ms=10):
    return TrajectoryStep(
        step_n=n, tool_called=tool,
        tool_args=args, tool_result=result,
        latency_ms=ms,
    )


def good_success_trace():
    return Trajectory(
        trace_id="demo-good-001",
        task_id="INV-DEMO-001",
        steps=(
            _step(1, "extract_fields", {"pdf_path": "data/INV-DEMO-001.pdf"},
                  {"text": "請求書 INV-DEMO-001...", "pages": 1, "char_count": 240}),
            _step(2, "validate",
                  {"subtotal": 350000, "consumption_tax": 35000, "total": 385000,
                   "invoice_number": "INV-DEMO-001"},
                  {"passed": True, "issues": [],
                   "details": {"subtotal": 350000,
                               "consumption_tax_stated": 35000,
                               "consumption_tax_expected": 35000,
                               "total_stated": 385000,
                               "total_expected": 385000}}),
            _step(3, "write_back",
                  {"subtotal": 350000, "consumption_tax": 35000, "total": 385000,
                   "invoice_number": "INV-DEMO-001"},
                  {"written": True, "output_path": "outputs/INV-DEMO-001.json"}),
        ),
        final_result={"status": "success"},
        total_steps=3,
    )


def bad_skip_validate_trace():
    return Trajectory(
        trace_id="demo-bad-skip",
        task_id="INV-BAD-SKIP",
        steps=(
            _step(1, "extract_fields", {"pdf_path": "data/INV-BAD.pdf"},
                  {"text": "請求書 INV-BAD...", "pages": 1}),
            _step(2, "write_back",
                  {"subtotal": 350000, "consumption_tax": 35000, "total": 385000,
                   "invoice_number": "INV-BAD"},
                  {"written": True, "output_path": "outputs/INV-BAD.json"}),
        ),
        final_result={"status": "success"},
        total_steps=2,
    )


def bad_writeback_after_failed_validate_trace():
    return Trajectory(
        trace_id="demo-bad-fail-wb",
        task_id="INV-BAD-FWBK",
        steps=(
            _step(1, "extract_fields", {"pdf_path": "data/INV-BAD2.pdf"},
                  {"text": "請求書 INV-BAD2...", "pages": 1}),
            _step(2, "validate",
                  {"subtotal": 250000, "consumption_tax": 30000, "total": 280000,
                   "invoice_number": "INV-BAD2"},
                  {"passed": False,
                   "issues": ["消費税の金額が不正: 記載値=30000円、正しい値=25000円"],
                   "details": {"subtotal": 250000,
                               "consumption_tax_stated": 30000,
                               "consumption_tax_expected": 25000,
                               "total_stated": 280000,
                               "total_expected": 280000}}),
            _step(3, "write_back",
                  {"subtotal": 250000, "consumption_tax": 30000, "total": 280000,
                   "invoice_number": "INV-BAD2"},
                  {"written": True, "output_path": "outputs/INV-BAD2.json"}),
        ),
        final_result={"status": "success"},
        total_steps=3,
    )


def bad_redundant_and_wrong_tool_trace():
    return Trajectory(
        trace_id="demo-bad-redundant",
        task_id="INV-BAD-REDUN",
        steps=(
            _step(1, "extract_fields", {"pdf_path": "data/INV-BAD3.pdf"},
                  {"text": "請求書 INV-BAD3...", "pages": 1}),
            _step(2, "extract_fields", {"pdf_path": "data/INV-BAD3.pdf"},
                  {"text": "請求書 INV-BAD3...", "pages": 1}),   # redundant!
            _step(3, "summarize_invoice", {"text": "請求書..."},
                  {"summary": "Invoice for 350000 yen"}),         # unexpected tool!
            _step(4, "validate",
                  {"subtotal": 350000, "consumption_tax": 35000, "total": 385000,
                   "invoice_number": "INV-BAD3"},
                  {"passed": True, "issues": []}),
            _step(5, "write_back",
                  {"subtotal": 350000, "consumption_tax": 35000, "total": 385000,
                   "invoice_number": "INV-BAD3"},
                  {"written": True, "output_path": "outputs/INV-BAD3.json"}),
        ),
        final_result={"status": "success"},
        total_steps=5,
    )


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------


def _separator(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("#" + " " * pad + title + " " * (width - pad - len(title) - 1) + "#")
    else:
        print("#" * width)


def scenario_good(label="A"):
    _separator(f"Scenario {label}: good trace (PASS expected)")
    rep = trajectory_report(good_success_trace(), SUCCESS_SPEC)
    print(rep.format())
    assert rep.passed
    return rep


def scenario_skip_validate(label="B"):
    _separator(f"Scenario {label}: skip validate → order violation (FAIL expected)")
    rep = trajectory_report(bad_skip_validate_trace(), SUCCESS_SPEC)
    print(rep.format())
    assert not rep.passed
    assert rep.order_violations
    return rep


def scenario_failed_validate_writeback(label="C"):
    _separator(f"Scenario {label}: write_back after failing validate (FAIL expected)")
    rep = trajectory_report(bad_writeback_after_failed_validate_trace(), SUCCESS_SPEC)
    print(rep.format())
    assert not rep.passed
    assert any("passed" in v.violation for v in rep.order_violations)
    return rep


def scenario_redundant_wrong_tool(label="D"):
    _separator(f"Scenario {label}: redundant call + unexpected tool (warnings, PASS)")
    # Redundant calls and unexpected tools surface as diagnostics. They do NOT
    # cause a hard FAIL on their own — add tools to forbidden_tools or tighten
    # max_steps in the spec to make them hard failures. The evaluator surfaces
    # them so you can decide the policy.
    rep = trajectory_report(bad_redundant_and_wrong_tool_trace(), SUCCESS_SPEC)
    print(rep.format())
    assert rep.step_efficiency.redundant_count == 1, "redundant call not detected"
    assert "summarize_invoice" in rep.tool_call_correctness.unexpected_tools
    # verdict is PASS because all expected tools are present, no violations,
    # step count exactly at limit — unexpected/redundant are warnings only.
    assert rep.passed
    return rep


def scenario_regression(label="E"):
    _separator(f"Scenario {label}: regression detection (ALARM expected)")
    good = [trajectory_report(good_success_trace(), SUCCESS_SPEC) for _ in range(8)]

    # Candidate: half skip validate (order violation + completion failure)
    bad_half = [
        trajectory_report(bad_skip_validate_trace(), SUCCESS_SPEC) for _ in range(4)
    ] + [
        trajectory_report(good_success_trace(), SUCCESS_SPEC) for _ in range(4)
    ]

    reg = detect_trajectory_regression(good, bad_half)
    print(reg.format())
    print()
    assert reg.alarm
    return reg


# ---------------------------------------------------------------------------
# Real-trace loader (optional, requires Phase A output path)
# ---------------------------------------------------------------------------


def _spec_for_real_trace(trace_path: Path) -> TrajectorySpec:
    import json
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    status = data.get("final_result", {}).get("status", "")
    return SUCCESS_SPEC if status == "success" else FLAGGED_SPEC


def scenario_real_traces(traces_dir: Path):
    _separator("Real Phase A traces from invoice-agent")
    from eval_sanity.trajectory import Trajectory

    trace_files = sorted(traces_dir.glob("*.json"))
    if not trace_files:
        print(f"  No JSON files found in {traces_dir}")
        return

    reports = []
    for tf in trace_files:
        traj = Trajectory.from_json(tf)
        spec = _spec_for_real_trace(tf)
        rep = trajectory_report(traj, spec)
        reports.append(rep)
        verdict = "PASS" if rep.passed else "FAIL"
        tc_info = ""
        if rep.task_completion:
            tc_info = f"  status={rep.task_completion.actual_status}"
        print(
            f"  {tf.name:<45} [{verdict}]"
            f"  steps={rep.step_efficiency.total_steps}"
            f"  violations={len(rep.order_violations)}"
            f"  redundant={rep.step_efficiency.redundant_count}"
            f"{tc_info}"
        )

    n_pass = sum(1 for r in reports if r.passed)
    print(f"\n  {n_pass}/{len(reports)} traces passed their spec.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    traces_dir: Path | None = None
    if "--traces-dir" in sys.argv:
        idx = sys.argv.index("--traces-dir")
        if idx + 1 < len(sys.argv):
            traces_dir = Path(sys.argv[idx + 1])

    scenario_good()
    print()
    scenario_skip_validate()
    print()
    scenario_failed_validate_writeback()
    print()
    scenario_redundant_wrong_tool()
    print()
    scenario_regression()

    if traces_dir is not None:
        print()
        scenario_real_traces(traces_dir)

    print()
    _separator("Summary")
    print("  All synthetic scenarios ran and assertions held.")
    print("  eval-sanity v0.3 — deterministic trajectory evaluation, zero LLM calls.")
    _separator()


if __name__ == "__main__":
    main()

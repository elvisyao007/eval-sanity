"""Deterministic trajectory evaluation tests.

Each test targets one decision branch. Bad-trajectory fixtures are hand-crafted
to trigger exactly the fault being tested, so the assertions prove the evaluator
has discrimination power — it catches each failure mode and only that one.

Fixture naming convention:
  _good_*     — trajectories that should pass the spec
  _bad_*      — trajectories that should fail in a specific way

Invoice-agent specs used throughout:
  SUCCESS_SPEC   — valid invoice: extract → validate(pass) → write_back
  FLAGGED_SPEC   — invalid invoice: extract → validate(fail), no write_back
"""

from eval_sanity import (
    OrderConstraint,
    TrajectorySpec,
    detect_trajectory_regression,
    trajectory_report,
)
from eval_sanity.trajectory import (
    Trajectory,
    TrajectoryStep,
)

# ---------------------------------------------------------------------------
# Shared specs
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
# Fixture helpers
# ---------------------------------------------------------------------------


def _step(n, tool, args, result, latency=10):
    return TrajectoryStep(
        step_n=n,
        tool_called=tool,
        tool_args=args,
        tool_result=result,
        latency_ms=latency,
    )


def _traj(task_id, steps, status, trace_id="trace-test"):
    return Trajectory(
        trace_id=trace_id,
        task_id=task_id,
        steps=tuple(steps),
        final_result={"status": status},
        total_steps=len(steps),
    )


# --- good fixtures ---

def _good_success():
    """Three-step success: extract → validate(pass) → write_back."""
    return _traj(
        "INV-001",
        [
            _step(1, "extract_fields", {"pdf_path": "inv.pdf"},
                  {"text": "請求書...", "pages": 1}),
            _step(2, "validate", {"subtotal": 350000},
                  {"passed": True, "issues": []}),
            _step(3, "write_back", {"subtotal": 350000},
                  {"written": True, "output_path": "outputs/INV-001.json"}),
        ],
        "success",
    )


def _good_flagged():
    """Two-step flagged: extract → validate(fail), no write_back."""
    return _traj(
        "INV-004",
        [
            _step(1, "extract_fields", {"pdf_path": "inv004.pdf"},
                  {"text": "請求書...", "pages": 1}),
            _step(2, "validate", {"subtotal": 250000, "consumption_tax": 30000},
                  {"passed": False, "issues": ["税額不正"]}),
        ],
        "flagged",
    )


# ---------------------------------------------------------------------------
# 1. Good trajectories pass their specs
# ---------------------------------------------------------------------------


def test_good_success_passes():
    rep = trajectory_report(_good_success(), SUCCESS_SPEC)
    assert rep.passed is True
    assert rep.tool_call_correctness.score == 1.0
    assert rep.tool_call_correctness.missing_tools == ()
    assert rep.tool_call_correctness.forbidden_tools_called == ()
    assert rep.order_violations == ()
    assert rep.step_efficiency.redundant_count == 0
    assert rep.task_completion is not None
    assert rep.task_completion.passed is True


def test_good_flagged_passes():
    rep = trajectory_report(_good_flagged(), FLAGGED_SPEC)
    assert rep.passed is True
    assert "validate" in rep.tool_call_correctness.called_tools
    assert "write_back" not in rep.tool_call_correctness.called_tools
    assert rep.tool_call_correctness.forbidden_tools_called == ()
    assert rep.order_violations == ()
    assert rep.task_completion.passed is True


# ---------------------------------------------------------------------------
# 2. Bad trajectory: skip validate, call write_back directly
# ---------------------------------------------------------------------------


def test_bad_skip_validate_order_violation():
    """write_back at step 2 with no preceding validate → order violation."""
    traj = _traj(
        "BAD-skip-validate",
        [
            _step(1, "extract_fields", {"pdf_path": "inv.pdf"},
                  {"text": "...", "pages": 1}),
            _step(2, "write_back", {"subtotal": 350000},
                  {"written": True, "output_path": "outputs/BAD.json"}),
        ],
        "success",
    )
    rep = trajectory_report(traj, SUCCESS_SPEC)

    assert rep.passed is False
    assert len(rep.order_violations) == 1
    v = rep.order_violations[0]
    assert v.tool == "write_back"
    assert v.step_n == 2
    assert "validate" in v.violation
    # Also missing validate in expected tools
    assert "validate" in rep.tool_call_correctness.missing_tools


def test_bad_skip_validate_not_a_false_negative():
    """Sanity check: the good trace does NOT trigger this violation."""
    rep = trajectory_report(_good_success(), SUCCESS_SPEC)
    assert rep.order_violations == ()


# ---------------------------------------------------------------------------
# 3. Bad trajectory: validate failed, write_back called anyway (buggy agent)
# ---------------------------------------------------------------------------


def test_bad_writeback_after_failed_validate():
    """validate returned passed=False, but write_back was called anyway."""
    traj = _traj(
        "BAD-write-on-fail",
        [
            _step(1, "extract_fields", {"pdf_path": "inv.pdf"},
                  {"text": "...", "pages": 1}),
            _step(2, "validate", {"subtotal": 250000, "consumption_tax": 30000},
                  {"passed": False, "issues": ["税額不正"]}),
            _step(3, "write_back", {"subtotal": 250000},
                  {"written": True, "output_path": "outputs/BAD.json"}),
        ],
        "success",
    )
    rep = trajectory_report(traj, SUCCESS_SPEC)

    assert rep.passed is False
    assert len(rep.order_violations) == 1
    v = rep.order_violations[0]
    assert v.tool == "write_back"
    assert "passed" in v.violation   # constraint mentions the pass field
    assert "never True" in v.violation


# ---------------------------------------------------------------------------
# 4. Bad trajectory: wrong / unexpected tool called
# ---------------------------------------------------------------------------


def test_bad_wrong_tool_detected():
    """An unknown tool 'summarize_invoice' appears — caught as unexpected."""
    traj = _traj(
        "BAD-wrong-tool",
        [
            _step(1, "extract_fields", {"pdf_path": "inv.pdf"},
                  {"text": "...", "pages": 1}),
            _step(2, "summarize_invoice", {"text": "..."},
                  {"summary": "..."}),
            _step(3, "validate", {"subtotal": 350000},
                  {"passed": True, "issues": []}),
            _step(4, "write_back", {"subtotal": 350000},
                  {"written": True, "output_path": "outputs/INV.json"}),
        ],
        "success",
    )
    rep = trajectory_report(traj, SUCCESS_SPEC)

    # All expected tools present, so score is 1.0 — but unexpected tool reported
    assert rep.tool_call_correctness.score == 1.0
    assert "summarize_invoice" in rep.tool_call_correctness.unexpected_tools
    assert "summarize_invoice" not in rep.tool_call_correctness.forbidden_tools_called
    # passed=False only if there are violations/missing/forbidden — unexpected alone
    # doesn't fail (it's a warning); passed depends on order/completion checks
    assert rep.order_violations == ()


def test_bad_forbidden_tool_write_back_on_flagged():
    """On a flagged trace, write_back is in forbidden_tools — fails hard."""
    traj = _traj(
        "BAD-forbidden-wb",
        [
            _step(1, "extract_fields", {"pdf_path": "inv004.pdf"},
                  {"text": "...", "pages": 1}),
            _step(2, "validate", {"subtotal": 250000},
                  {"passed": False, "issues": ["税額不正"]}),
            _step(3, "write_back", {"subtotal": 250000},
                  {"written": True, "output_path": "outputs/BAD.json"}),
        ],
        "flagged",
    )
    rep = trajectory_report(traj, FLAGGED_SPEC)

    assert rep.passed is False
    assert "write_back" in rep.tool_call_correctness.forbidden_tools_called
    # forbidden tools are NOT in unexpected_tools (separate bucket)
    assert "write_back" not in rep.tool_call_correctness.unexpected_tools


# ---------------------------------------------------------------------------
# 5. Bad trajectory: redundant calls (same tool, same args repeated)
# ---------------------------------------------------------------------------


def test_bad_redundant_extract_detected():
    """extract_fields called twice with identical args → redundant_count=1."""
    traj = _traj(
        "BAD-redundant",
        [
            _step(1, "extract_fields", {"pdf_path": "inv.pdf"},
                  {"text": "...", "pages": 1}),
            _step(2, "extract_fields", {"pdf_path": "inv.pdf"},
                  {"text": "...", "pages": 1}),
            _step(3, "validate", {"subtotal": 350000},
                  {"passed": True, "issues": []}),
            _step(4, "write_back", {"subtotal": 350000},
                  {"written": True, "output_path": "outputs/INV.json"}),
        ],
        "success",
    )
    rep = trajectory_report(traj, SUCCESS_SPEC)

    assert rep.step_efficiency.redundant_count == 1
    assert len(rep.step_efficiency.redundant_calls) == 1
    rc = rep.step_efficiency.redundant_calls[0]
    assert rc.tool == "extract_fields"
    assert rc.step_ns == (1, 2)


def test_different_args_not_redundant():
    """Two extract_fields calls with DIFFERENT args are not redundant."""
    traj = _traj(
        "GOOD-two-extracts",
        [
            _step(1, "extract_fields", {"pdf_path": "inv001.pdf"},
                  {"text": "...", "pages": 1}),
            _step(2, "extract_fields", {"pdf_path": "inv002.pdf"},
                  {"text": "...", "pages": 1}),
            _step(3, "validate", {"subtotal": 350000},
                  {"passed": True, "issues": []}),
            _step(4, "write_back", {"subtotal": 350000},
                  {"written": True, "output_path": "outputs/INV.json"}),
        ],
        "success",
    )
    rep = trajectory_report(traj, SUCCESS_SPEC)
    assert rep.step_efficiency.redundant_count == 0


# ---------------------------------------------------------------------------
# 6. Task completion mismatch
# ---------------------------------------------------------------------------


def test_bad_wrong_final_status():
    """Trace says 'success' but spec expects 'flagged' → task_completion fails."""
    traj = _good_success()   # final_result.status = "success"
    rep = trajectory_report(traj, FLAGGED_SPEC)

    assert rep.task_completion is not None
    assert rep.task_completion.passed is False
    assert rep.task_completion.actual_status == "success"
    assert rep.task_completion.expected_status == "flagged"
    assert rep.passed is False


def test_no_task_completion_check_when_spec_omits_status():
    spec = TrajectorySpec(
        expected_tools=["extract_fields", "validate", "write_back"],
    )
    rep = trajectory_report(_good_success(), spec)
    assert rep.task_completion is None


# ---------------------------------------------------------------------------
# 7. Step budget
# ---------------------------------------------------------------------------


def test_exceeds_max_steps():
    """A 3-step trace against a spec with max_steps=2 flags exceeds_max."""
    spec = TrajectorySpec(
        expected_tools=["extract_fields"],
        max_steps=2,
    )
    traj = _traj(
        "OVER-budget",
        [
            _step(1, "extract_fields", {}, {}),
            _step(2, "validate", {}, {}),
            _step(3, "write_back", {}, {}),
        ],
        "success",
    )
    rep = trajectory_report(traj, spec)
    assert rep.step_efficiency.exceeds_max is True
    assert rep.passed is False


# ---------------------------------------------------------------------------
# 8. from_dict / from_json round-trip
# ---------------------------------------------------------------------------


def test_trajectory_from_dict():
    d = {
        "trace_id": "abc",
        "invoice_id": "INV-999",
        "model": "test-model",
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:05Z",
        "total_steps": 2,
        "total_latency_ms": 5000,
        "steps": [
            {
                "step_n": 1,
                "tool_called": "extract_fields",
                "tool_args": {"pdf_path": "x.pdf"},
                "tool_result": {"text": "abc", "pages": 1},
                "latency_ms": 10,
                "llm_latency_ms": 500,
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "step_n": 2,
                "tool_called": "validate",
                "tool_args": {"subtotal": 100},
                "tool_result": {"passed": True, "issues": []},
                "latency_ms": 0,
            },
        ],
        "final_result": {"status": "flagged"},
        "llm_calls": 3,
    }
    traj = Trajectory.from_dict(d)

    assert traj.trace_id == "abc"
    assert traj.task_id == "INV-999"
    assert traj.model == "test-model"
    assert traj.total_steps == 2
    assert len(traj.steps) == 2
    assert traj.steps[0].tool_called == "extract_fields"
    assert traj.steps[1].tool_result["passed"] is True
    assert traj.final_result["status"] == "flagged"


def test_trajectory_from_json(tmp_path):
    import json

    data = {
        "trace_id": "t1",
        "task_id": "T-1",
        "steps": [
            {"step_n": 1, "tool_called": "extract_fields",
             "tool_args": {}, "tool_result": {}, "latency_ms": 5}
        ],
        "final_result": {"status": "success"},
        "total_steps": 1,
    }
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    traj = Trajectory.from_json(p)
    assert traj.trace_id == "t1"
    assert traj.task_id == "T-1"
    assert len(traj.steps) == 1


# ---------------------------------------------------------------------------
# 9. TrajectoryReport.format() smoke test
# ---------------------------------------------------------------------------


def test_format_pass():
    rep = trajectory_report(_good_success(), SUCCESS_SPEC)
    text = rep.format()
    assert "PASS" in text
    assert "trajectory matches spec" in text
    assert str(rep) == text


def test_format_fail_skip_validate():
    traj = _traj(
        "FMT",
        [
            _step(1, "extract_fields", {"pdf_path": "x.pdf"}, {}),
            _step(2, "write_back", {"subtotal": 100}, {"written": True}),
        ],
        "success",
    )
    rep = trajectory_report(traj, SUCCESS_SPEC)
    text = rep.format()
    assert "FAIL" in text
    assert "order violation" in text.lower() or "order_violations" in text


# ---------------------------------------------------------------------------
# 10. detect_trajectory_regression
# ---------------------------------------------------------------------------


def test_regression_alarms_on_completion_drop():
    """Baseline: all pass. Candidate: half fail. Should alarm."""
    good_reports = [
        trajectory_report(_good_success(), SUCCESS_SPEC) for _ in range(10)
    ]

    # Build a "bad" report by applying success spec to a skip-validate trace
    def _bad():
        t = _traj(
            "BAD",
            [
                _step(1, "extract_fields", {}, {}),
                _step(2, "write_back", {}, {"written": True}),
            ],
            "success",
        )
        return trajectory_report(t, SUCCESS_SPEC)

    bad_reports = [_bad() for _ in range(10)]

    reg = detect_trajectory_regression(good_reports, bad_reports)
    assert reg.alarm is True
    assert "completion_rate" in reg.verdict
    assert reg.completion_rate_delta < 0


def test_regression_alarms_on_violation_rate_increase():
    """Candidate has 100% order violations; baseline has 0%."""
    good = [trajectory_report(_good_success(), SUCCESS_SPEC) for _ in range(5)]

    def _violation_trace():
        t = _traj(
            "VIO",
            [
                _step(1, "extract_fields", {}, {}),
                _step(2, "validate", {}, {"passed": False}),
                _step(3, "write_back", {}, {"written": True}),
            ],
            "success",
        )
        return trajectory_report(t, SUCCESS_SPEC)

    bad = [_violation_trace() for _ in range(5)]
    reg = detect_trajectory_regression(good, bad)
    assert reg.alarm is True
    assert reg.violation_rate_delta > 0


def test_regression_stable_no_alarm():
    """Both sets identical — no alarm."""
    reports = [trajectory_report(_good_success(), SUCCESS_SPEC) for _ in range(6)]
    reg = detect_trajectory_regression(reports, reports)
    assert reg.alarm is False
    assert reg.completion_rate_delta == 0.0
    assert reg.mean_steps_delta == 0.0


def test_regression_alarms_on_step_increase():
    """Candidate takes many more steps than baseline."""
    def _short():
        t = _traj("S", [_step(1, "extract_fields", {}, {})], "success")
        return trajectory_report(t, TrajectorySpec(expected_tools=["extract_fields"]))

    def _long():
        steps = [_step(i + 1, "extract_fields" if i == 0 else f"extra_{i}", {}, {})
                 for i in range(6)]
        t = _traj("L", steps, "success", trace_id=f"tr-{id(steps)}")
        return trajectory_report(t, TrajectorySpec(expected_tools=["extract_fields"]))

    short_reports = [_short() for _ in range(5)]
    long_reports = [_long() for _ in range(5)]
    reg = detect_trajectory_regression(
        short_reports, long_reports, step_increase_threshold=2.0
    )
    assert reg.alarm is True
    assert reg.mean_steps_delta > 2.0


def test_regression_report_format():
    good = [trajectory_report(_good_success(), SUCCESS_SPEC) for _ in range(4)]
    bad = [
        trajectory_report(
            _traj("B", [_step(1, "extract_fields", {}, {}), _step(2, "write_back", {}, {})],
                  "success"),
            SUCCESS_SPEC,
        )
        for _ in range(4)
    ]
    reg = detect_trajectory_regression(good, bad)
    text = reg.format()
    assert "trajectory-regression" in text
    assert "completion_rate" in text
    assert str(reg) == text


def test_regression_empty_candidate():
    """Empty candidate set — no alarm (nothing to compare, n=0 candidate)."""
    good = [trajectory_report(_good_success(), SUCCESS_SPEC) for _ in range(3)]
    reg = detect_trajectory_regression(good, [])
    # completion_rate drops from 1.0 to 0.0 — should alarm
    assert reg.n_candidate == 0
    # depends on threshold: 0.0 - 1.0 = -1.0, clearly > 0.05 drop → alarm
    assert reg.alarm is True

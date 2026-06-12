"""Deterministic agent trajectory evaluation (v0.3).

Audits whether an agent's tool-call sequence satisfies a declarative spec:
  - tool_call_correctness: required tools called? forbidden tools avoided?
  - order_constraint_check: precedence rules hold (e.g. write_back only after
    passing validate)?
  - step_efficiency: step count within budget; redundant (same-tool same-args)
    calls flagged.
  - task_completion: final_result status matches expected?

All checks are deterministic — no LLM calls, no external dependencies. Extends
eval-sanity's zero-dependency, zero-randomness philosophy from retrieval metrics
to agent trajectory audits.

detect_trajectory_regression compares two sets of TrajectoryReport objects
(e.g. baseline vs. new prompt / new model) and flags when completion rate, step
count, or violation rate degrades in the candidate set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model — matches Phase A / invoice-agent trace JSON schema
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    """One tool invocation recorded in an agent trace."""

    step_n: int
    tool_called: str
    tool_args: dict
    tool_result: dict
    latency_ms: int = 0
    llm_latency_ms: int | None = None
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryStep:
        return cls(
            step_n=int(d["step_n"]),
            tool_called=str(d["tool_called"]),
            tool_args=dict(d.get("tool_args") or {}),
            tool_result=dict(d.get("tool_result") or {}),
            latency_ms=int(d.get("latency_ms", 0)),
            llm_latency_ms=d.get("llm_latency_ms"),
            timestamp=d.get("timestamp"),
        )


@dataclass
class Trajectory:
    """Full agent run: envelope metadata plus ordered tool-call sequence."""

    trace_id: str
    task_id: str
    steps: tuple[TrajectoryStep, ...]
    final_result: dict
    total_steps: int
    model: str | None = None
    total_latency_ms: int | None = None
    started_at: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Trajectory:
        steps = tuple(TrajectoryStep.from_dict(s) for s in d.get("steps", []))
        return cls(
            trace_id=str(d.get("trace_id", "")),
            task_id=str(d.get("task_id") or d.get("invoice_id", "")),
            steps=steps,
            final_result=dict(d.get("final_result") or {}),
            total_steps=int(d.get("total_steps", len(steps))),
            model=d.get("model"),
            total_latency_ms=d.get("total_latency_ms"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Trajectory:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Spec — declares what a correct trajectory looks like
# ---------------------------------------------------------------------------


@dataclass
class OrderConstraint:
    """Enforce: every call to ``tool`` must be preceded by ``required_predecessor``.

    If ``require_pass_field`` is set (e.g. ``"passed"``), the predecessor's
    ``tool_result`` must contain that field equal to ``True`` for the constraint
    to be satisfied. Useful for guarding write-back tools behind validation steps.
    """

    tool: str
    required_predecessor: str
    require_pass_field: str | None = None


@dataclass
class TrajectorySpec:
    """Declarative specification for a correct agent trajectory.

    Attributes:
        expected_tools: Tool names that must appear at least once.
        forbidden_tools: Tool names that must never appear.
        order_constraints: Precedence rules (see :class:`OrderConstraint`).
        max_steps: Upper bound on total tool calls; ``None`` means no limit.
        expected_final_status: If set, checked against ``final_result["status"]``.
    """

    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    order_constraints: list[OrderConstraint] = field(default_factory=list)
    max_steps: int | None = None
    expected_final_status: str | None = None


# ---------------------------------------------------------------------------
# Result dataclasses (all frozen — deterministic, safe to compare)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallCorrectness:
    """Whether required tools were called and forbidden ones were avoided.

    ``unexpected_tools`` are calls not in ``expected_tools`` and not explicitly
    forbidden — they are "unrecognised" rather than prohibited. Check
    ``forbidden_tools_called`` for hard failures.
    """

    expected_tools: tuple[str, ...]
    called_tools: tuple[str, ...]         # unique tools that appeared, first-seen order
    missing_tools: tuple[str, ...]        # in expected but never called
    unexpected_tools: tuple[str, ...]     # called, not in expected, not forbidden
    forbidden_tools_called: tuple[str, ...]  # explicitly forbidden and called
    score: float                          # fraction of expected_tools present [0..1]


@dataclass(frozen=True)
class OrderViolation:
    """One instance of a precedence constraint being broken."""

    tool: str
    step_n: int
    violation: str


@dataclass(frozen=True)
class RedundantCall:
    """Identical (same tool, same args) calls that waste agent steps."""

    tool: str
    step_ns: tuple[int, ...]   # all step numbers where this call occurred
    args_repr: str              # JSON of tool_args, for human inspection


@dataclass(frozen=True)
class StepEfficiency:
    """Step count and redundancy analysis."""

    total_steps: int
    max_steps: int | None
    exceeds_max: bool
    redundant_calls: tuple[RedundantCall, ...]
    redundant_count: int   # number of *extra* (duplicate) invocations across all groups


@dataclass(frozen=True)
class TaskCompletion:
    """Whether ``final_result["status"]`` matched the expected status."""

    expected_status: str
    actual_status: str
    passed: bool


@dataclass(frozen=True)
class TrajectoryReport:
    """Full deterministic audit of one agent trajectory against a spec.

    ``passed`` is ``True`` only when every check passes: all expected tools
    called, no forbidden tools, no order violations, step count within budget,
    and (if specified) the task completed with the expected status.
    """

    trace_id: str
    task_id: str
    tool_call_correctness: ToolCallCorrectness
    order_violations: tuple[OrderViolation, ...]
    step_efficiency: StepEfficiency
    task_completion: TaskCompletion | None
    passed: bool

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _verdict_line(self) -> str:
        if self.passed:
            return "trajectory matches spec — no violations"
        parts: list[str] = []
        tcc = self.tool_call_correctness
        if tcc.missing_tools:
            parts.append(f"missing tools {list(tcc.missing_tools)}")
        if tcc.forbidden_tools_called:
            parts.append(f"forbidden tools called {list(tcc.forbidden_tools_called)}")
        if self.order_violations:
            parts.append(f"{len(self.order_violations)} order violation(s)")
        if self.step_efficiency.exceeds_max:
            parts.append(f"exceeded max_steps ({self.step_efficiency.max_steps})")
        if self.task_completion is not None and not self.task_completion.passed:
            tc = self.task_completion
            parts.append(
                f"wrong final status '{tc.actual_status}' "
                f"(expected '{tc.expected_status}')"
            )
        return "; ".join(parts) if parts else "unknown failure"

    def format(self) -> str:
        lines = [
            f"trajectory-report  trace={self.trace_id}  task={self.task_id}",
            "=" * 70,
        ]
        tcc = self.tool_call_correctness
        lines += [
            f"  tool_call_correctness  score={tcc.score:.2f}",
            f"    expected : {list(tcc.expected_tools)}",
            f"    called   : {list(tcc.called_tools)}",
        ]
        if tcc.missing_tools:
            lines.append(f"    MISSING  : {list(tcc.missing_tools)}")
        if tcc.forbidden_tools_called:
            lines.append(f"    FORBIDDEN: {list(tcc.forbidden_tools_called)}")
        if tcc.unexpected_tools:
            lines.append(f"    unexpected (not in spec): {list(tcc.unexpected_tools)}")

        eff = self.step_efficiency
        step_line = f"  step_efficiency  steps={eff.total_steps}"
        if eff.max_steps is not None:
            flag = "  EXCEEDS_MAX" if eff.exceeds_max else ""
            step_line += f" (max={eff.max_steps}{flag})"
        step_line += f"  redundant_extra_calls={eff.redundant_count}"
        lines.append(step_line)
        for rc in eff.redundant_calls:
            lines.append(f"    redundant: '{rc.tool}' called at steps {list(rc.step_ns)}")

        if self.order_violations:
            lines.append(f"  order_violations  ({len(self.order_violations)})")
            for v in self.order_violations:
                lines.append(f"    step {v.step_n} '{v.tool}': {v.violation}")
        else:
            lines.append("  order_violations  (0)  — none")

        if self.task_completion is not None:
            tc = self.task_completion
            status_tag = "PASS" if tc.passed else "FAIL"
            lines.append(
                f"  task_completion  expected='{tc.expected_status}'"
                f"  actual='{tc.actual_status}'  [{status_tag}]"
            )

        lines.append("=" * 70)
        verdict_tag = "PASS" if self.passed else "FAIL"
        lines.append(f"  [{verdict_tag}]  {self._verdict_line()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def trajectory_report(trajectory: Trajectory, spec: TrajectorySpec) -> TrajectoryReport:
    """Evaluate *trajectory* deterministically against *spec*.

    No LLM calls. No external dependencies. All results are bit-for-bit identical
    across runs given the same inputs.

    Args:
        trajectory: The agent run to evaluate.
        spec: The declarative specification it should satisfy.

    Returns:
        A :class:`TrajectoryReport` with per-check results and an overall pass/fail.
    """
    steps = trajectory.steps

    # ------------------------------------------------------------------
    # 1. Tool call correctness
    # ------------------------------------------------------------------
    called_in_order: list[str] = []
    seen_tools: set[str] = set()
    for s in steps:
        if s.tool_called not in seen_tools:
            called_in_order.append(s.tool_called)
            seen_tools.add(s.tool_called)

    expected_set = set(spec.expected_tools)
    forbidden_set = set(spec.forbidden_tools)

    missing = [t for t in spec.expected_tools if t not in seen_tools]
    forbidden_called = [t for t in called_in_order if t in forbidden_set]
    unexpected = [t for t in called_in_order if t not in expected_set and t not in forbidden_set]

    score = (
        sum(1 for t in spec.expected_tools if t in seen_tools) / len(spec.expected_tools)
        if spec.expected_tools
        else 1.0
    )

    tcc = ToolCallCorrectness(
        expected_tools=tuple(spec.expected_tools),
        called_tools=tuple(called_in_order),
        missing_tools=tuple(missing),
        unexpected_tools=tuple(unexpected),
        forbidden_tools_called=tuple(forbidden_called),
        score=score,
    )

    # ------------------------------------------------------------------
    # 2. Order constraint check
    # ------------------------------------------------------------------
    violations: list[OrderViolation] = []
    for constraint in spec.order_constraints:
        for i, step in enumerate(steps):
            if step.tool_called != constraint.tool:
                continue
            preceding = steps[:i]
            predecessors = [s for s in preceding if s.tool_called == constraint.required_predecessor]
            if not predecessors:
                violations.append(
                    OrderViolation(
                        tool=step.tool_called,
                        step_n=step.step_n,
                        violation=f"no preceding '{constraint.required_predecessor}' call",
                    )
                )
            elif constraint.require_pass_field is not None:
                has_passing = any(
                    s.tool_result.get(constraint.require_pass_field) is True
                    for s in predecessors
                )
                if not has_passing:
                    violations.append(
                        OrderViolation(
                            tool=step.tool_called,
                            step_n=step.step_n,
                            violation=(
                                f"preceded only by failing '{constraint.required_predecessor}'"
                                f" ('{constraint.require_pass_field}' never True)"
                            ),
                        )
                    )

    # ------------------------------------------------------------------
    # 3. Step efficiency — redundancy detection
    # ------------------------------------------------------------------
    args_groups: dict[str, list[int]] = {}
    for step in steps:
        try:
            args_key = json.dumps(step.tool_args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_key = repr(step.tool_args)
        group_key = f"{step.tool_called}\x00{args_key}"
        args_groups.setdefault(group_key, []).append(step.step_n)

    redundant_groups: list[RedundantCall] = []
    total_redundant_extra = 0
    for group_key, step_ns in args_groups.items():
        if len(step_ns) > 1:
            tool_name, args_repr = group_key.split("\x00", 1)
            redundant_groups.append(
                RedundantCall(
                    tool=tool_name,
                    step_ns=tuple(step_ns),
                    args_repr=args_repr,
                )
            )
            total_redundant_extra += len(step_ns) - 1

    exceeds_max = spec.max_steps is not None and trajectory.total_steps > spec.max_steps
    eff = StepEfficiency(
        total_steps=trajectory.total_steps,
        max_steps=spec.max_steps,
        exceeds_max=exceeds_max,
        redundant_calls=tuple(redundant_groups),
        redundant_count=total_redundant_extra,
    )

    # ------------------------------------------------------------------
    # 4. Task completion
    # ------------------------------------------------------------------
    tc: TaskCompletion | None = None
    if spec.expected_final_status is not None:
        actual_status = str(trajectory.final_result.get("status", ""))
        tc = TaskCompletion(
            expected_status=spec.expected_final_status,
            actual_status=actual_status,
            passed=(actual_status == spec.expected_final_status),
        )

    # ------------------------------------------------------------------
    # 5. Overall pass/fail
    # ------------------------------------------------------------------
    passed = (
        not tcc.missing_tools
        and not tcc.forbidden_tools_called
        and not violations
        and not exceeds_max
        and (tc is None or tc.passed)
    )

    return TrajectoryReport(
        trace_id=trajectory.trace_id,
        task_id=trajectory.task_id,
        tool_call_correctness=tcc,
        order_violations=tuple(violations),
        step_efficiency=eff,
        task_completion=tc,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Trajectory regression detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectorySetStats:
    """Aggregate statistics for a set of :class:`TrajectoryReport` objects."""

    n: int
    completion_rate: float   # fraction with correct final status (or overall passed)
    mean_steps: float        # average total_steps
    redundancy_rate: float   # fraction with >= 1 redundant call
    violation_rate: float    # fraction with >= 1 order violation


@dataclass(frozen=True)
class TrajectoryRegressionReport:
    """Comparison of two trajectory sets: baseline vs candidate.

    ``alarm`` fires when completion_rate drops, step count rises, or violation
    rate increases beyond the configured thresholds.
    """

    n_baseline: int
    n_candidate: int
    baseline: TrajectorySetStats
    candidate: TrajectorySetStats
    completion_rate_delta: float   # candidate - baseline (negative = regression)
    mean_steps_delta: float        # candidate - baseline (positive = slower)
    redundancy_rate_delta: float   # candidate - baseline (positive = more waste)
    violation_rate_delta: float    # candidate - baseline (positive = more violations)
    alarm: bool
    verdict: str

    def format(self) -> str:
        b, c = self.baseline, self.candidate
        lines = [
            f"trajectory-regression  baseline_n={self.n_baseline}"
            f"  candidate_n={self.n_candidate}",
            "=" * 70,
            f"  completion_rate : {b.completion_rate:.3f} -> {c.completion_rate:.3f}"
            f"  ({self.completion_rate_delta:+.3f})",
            f"  mean_steps      : {b.mean_steps:.1f} -> {c.mean_steps:.1f}"
            f"  ({self.mean_steps_delta:+.1f})",
            f"  redundancy_rate : {b.redundancy_rate:.3f} -> {c.redundancy_rate:.3f}"
            f"  ({self.redundancy_rate_delta:+.3f})",
            f"  violation_rate  : {b.violation_rate:.3f} -> {c.violation_rate:.3f}"
            f"  ({self.violation_rate_delta:+.3f})",
            "=" * 70,
            f"  {'*** ALARM *** ' if self.alarm else ''}{self.verdict}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def _trajectory_set_stats(reports: list[TrajectoryReport]) -> TrajectorySetStats:
    n = len(reports)
    if n == 0:
        return TrajectorySetStats(
            n=0,
            completion_rate=0.0,
            mean_steps=0.0,
            redundancy_rate=0.0,
            violation_rate=0.0,
        )

    return TrajectorySetStats(
        n=n,
        completion_rate=sum(1 for r in reports if r.passed) / n,
        mean_steps=sum(r.step_efficiency.total_steps for r in reports) / n,
        redundancy_rate=sum(1 for r in reports if r.step_efficiency.redundant_count > 0) / n,
        violation_rate=sum(1 for r in reports if r.order_violations) / n,
    )


def detect_trajectory_regression(
    baseline: list[TrajectoryReport],
    candidate: list[TrajectoryReport],
    *,
    completion_drop_threshold: float = 0.05,
    step_increase_threshold: float = 1.0,
    violation_increase_threshold: float = 0.10,
) -> TrajectoryRegressionReport:
    """Compare two sets of trajectory reports and flag regressions.

    Alarm fires when any of the following cross a threshold in the degrading
    direction:

    - ``completion_rate`` drops by more than *completion_drop_threshold* (default 5%).
    - ``mean_steps`` rises by more than *step_increase_threshold* (default 1 step).
    - ``violation_rate`` rises by more than *violation_increase_threshold* (default 10%).

    For small trajectory sets (n < 30), paired-bootstrap CIs are not meaningful.
    Thresholds are the primary guard. Extend with bootstrap CIs (see
    :func:`eval_sanity.regression.detect_regression`) when n is large enough to
    carry statistical power.

    Args:
        baseline: Reports from the reference run (previous prompt / model).
        candidate: Reports from the run under test.
        completion_drop_threshold: Alarm when completion_rate drops by this much.
        step_increase_threshold: Alarm when mean_steps rises by this much.
        violation_increase_threshold: Alarm when violation_rate rises by this much.

    Returns:
        A :class:`TrajectoryRegressionReport` with alarm flag and verdict.
    """
    base_stats = _trajectory_set_stats(baseline)
    cand_stats = _trajectory_set_stats(candidate)

    cr_delta = cand_stats.completion_rate - base_stats.completion_rate
    st_delta = cand_stats.mean_steps - base_stats.mean_steps
    red_delta = cand_stats.redundancy_rate - base_stats.redundancy_rate
    viol_delta = cand_stats.violation_rate - base_stats.violation_rate

    reasons: list[str] = []
    if cr_delta < -completion_drop_threshold:
        reasons.append(
            f"completion_rate dropped {cr_delta:+.3f} (threshold -{completion_drop_threshold:.2f})"
        )
    if st_delta > step_increase_threshold:
        reasons.append(
            f"mean_steps rose {st_delta:+.1f} (threshold +{step_increase_threshold:.1f})"
        )
    if viol_delta > violation_increase_threshold:
        reasons.append(
            f"violation_rate rose {viol_delta:+.3f} (threshold +{violation_increase_threshold:.2f})"
        )

    alarm = bool(reasons)
    verdict = (
        "TRAJECTORY REGRESSION detected: " + "; ".join(reasons)
        if alarm
        else "no significant trajectory regression (all metrics within thresholds)"
    )

    return TrajectoryRegressionReport(
        n_baseline=len(baseline),
        n_candidate=len(candidate),
        baseline=base_stats,
        candidate=cand_stats,
        completion_rate_delta=cr_delta,
        mean_steps_delta=st_delta,
        redundancy_rate_delta=red_delta,
        violation_rate_delta=viol_delta,
        alarm=alarm,
        verdict=verdict,
    )

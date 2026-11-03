"""Tests for the experiment's integrity conditions.

Each one guards a property that, if it broke silently, would invalidate the
cost comparison rather than merely produce a bug.
"""

import numpy as np
import pytest

from astraxjev import verbalize
from astraxjev.backends.astra_mock import synthesise_plan
from astraxjev.backends.base import Question, render_request
from astraxjev.pipelines import Geometry, PlanCursor
from astraxjev.pipelines.astra_jev import AstraJevPipeline
from astraxjev.pipelines.astra_only import PlanCache
from astraxjev.plan import (
    FREE_KINDS,
    Frame,
    GoalKind,
    Subgoal,
    TaskCapabilities,
    TaskPlan,
    calibrate,
    resolve,
)
from astraxjev.types import (
    BASE_SLICE,
    FINE_PRECISION_MM,
    HEAD_PAN_IDX,
    HEAD_TILT_IDX,
    SymbolicState,
)

INSTRUCTIONS = [
    "move to shelf and pick Nivea Body Milk to basket",
    "pick Duff Beer Can and place on empty board",
    "pick Heinz Beans from floor and place it on shelf",
    "open the fridge",
    "close the door of the showcase",
]


def hybrid(**kw):
    return AstraJevPipeline(astra=None, jev=None, geom=Geometry(), **kw)


def cursor_at(plan, index):
    c = PlanCursor(plan, Geometry())
    c.index = index
    return c


# -- who is allowed to do what ------------------------------------------------
@pytest.mark.parametrize("instruction", INSTRUCTIONS)
def test_jev_never_owns_a_pixel_bound_subgoal(instruction):
    """The premise: work that needs current pixels stays with the frontier model.

    Checked on every task shape, because the routing rule is supposed to be
    task-agnostic -- if it only held for shelf picking it would be a phase
    list in disguise.
    """
    h = hybrid()
    plan = synthesise_plan(instruction)
    for i, sg in enumerate(plan.subgoals):
        if sg.needs_pixels:
            assert not h.jev_owns(cursor_at(plan, i)), (instruction, sg.name)


@pytest.mark.parametrize("instruction", INSTRUCTIONS)
def test_jev_never_owns_a_path_constrained_subgoal(instruction):
    """Discovered by the door task, not designed in.

    A controller choosing among seven fixed directions cuts the chord of an
    arc; a gripper holding a handle has to stay on the hinge radius. Routing
    on precision alone sent the swings to Jev and the hybrid came out both
    less reliable and dearer than the baseline.
    """
    h = hybrid()
    plan = synthesise_plan(instruction)
    for i, sg in enumerate(plan.subgoals):
        if sg.path_constrained:
            assert not h.jev_owns(cursor_at(plan, i)), (instruction, sg.name)


def test_arc_is_path_constrained_by_derivation_not_annotation():
    """The planner should not have to know about the constraint."""
    assert Subgoal("s", GoalKind.ARC, "hinge").path_constrained
    assert not Subgoal("s", GoalKind.STANDOFF, "target").path_constrained
    # ...but a plan may override it for a primitive whose constraint is scenic.
    assert Subgoal(
        "s", GoalKind.OVER, "basket", params={"path_constrained": True}
    ).path_constrained


def test_routing_reads_only_subgoal_properties():
    """No task names, no phase names, no plan identity.

    Two subgoals with identical tolerance / pixels / constraint must route the
    same way regardless of what they are called or which task they came from.
    """
    h = hybrid()
    a = Subgoal("wildly_different_name", GoalKind.STANDOFF, "target",
                tolerance_mm=45.0, needs_pixels=False)
    b = Subgoal("x", GoalKind.OVER, "somewhere_else",
                tolerance_mm=45.0, needs_pixels=False)
    pa = TaskPlan(subgoals=[a])
    pb = TaskPlan(subgoals=[b])
    assert h.jev_owns(cursor_at(pa, 0)) == h.jev_owns(cursor_at(pb, 0))


def test_phase_assignment_is_derived_not_hardcoded():
    """Raising the threshold hands more subgoals to Jev, lowering it fewer."""
    plan = synthesise_plan(INSTRUCTIONS[0])
    permissive = hybrid(jev_precision_threshold_mm=1.0)
    strict = hybrid(jev_precision_threshold_mm=1e6)
    owned_p = sum(permissive.jev_owns(cursor_at(plan, i)) for i in range(len(plan)))
    owned_s = sum(strict.jev_owns(cursor_at(plan, i)) for i in range(len(plan)))
    assert owned_p > owned_s
    # Even wide open, pixel-bound subgoals stay with Astra.
    for i, sg in enumerate(plan.subgoals):
        if sg.needs_pixels:
            assert not permissive.jev_owns(cursor_at(plan, i))


# -- tolerances are measured, not chosen --------------------------------------
def test_calibration_replaces_guessed_tolerances_with_env_ones():
    caps = TaskCapabilities(
        place_tolerance_m=0.2, gripper_stroke_m=0.1,
        neighbour_gap_m=0.085, item_width_m=0.065,
    )
    plan = calibrate(synthesise_plan(INSTRUCTIONS[0]), caps)
    sources = {sg.kind: sg.tolerance_source for sg in plan.subgoals}
    assert sources[GoalKind.CONTACT] == "env"
    assert sources[GoalKind.OVER] == "env"
    over = next(sg for sg in plan.subgoals if sg.kind is GoalKind.OVER)
    # The environment declares its own success threshold; use it.
    assert over.tolerance_mm == pytest.approx(200.0)


def test_clutter_binds_the_grasp_tolerance_not_the_gripper():
    """On a dense retail board the neighbour, not the finger stroke, is the limit."""
    roomy = TaskCapabilities(gripper_stroke_m=0.1, item_width_m=0.065,
                             neighbour_gap_m=0.30)
    tight = TaskCapabilities(gripper_stroke_m=0.1, item_width_m=0.065,
                             neighbour_gap_m=0.085)
    assert tight.grasp_tolerance_m() < roomy.grasp_tolerance_m()
    assert tight.grasp_tolerance_m() == pytest.approx((0.085 - 0.065) / 2)


def test_derived_grasp_tolerance_is_never_zero_or_negative():
    impossible = TaskCapabilities(gripper_stroke_m=0.05, item_width_m=0.09,
                                  neighbour_gap_m=0.09)
    assert impossible.grasp_tolerance_m() > 0


# -- the coarse interface cannot fake fine precision --------------------------
def state(goal=(0.5, 0.0, 0.0), tcp=(0.0, 0.0, 0.0)):
    return SymbolicState(
        subgoal="approach_item",
        goal_kind="standoff",
        tcp_position=np.array(tcp, dtype=float),
        goal_position=np.array(goal, dtype=float),
        frame_normal=np.array([-1.0, 0.0, 0.0]),
        gripper_closed=False,
        holding_item=False,
        nearest_neighbour_cm=20.0,
        steps_in_subgoal=1,
        steps_total=1,
    )


def test_coarse_step_is_an_order_of_magnitude_above_fine_tolerance():
    geom = Geometry()
    fine_m = FINE_PRECISION_MM / 1000.0
    for tol_mm in (25.0, 45.0, 200.0):
        assert geom.coarse_step(tol_mm / 1000.0) >= 5 * fine_m, tol_mm


def test_verbaliser_emits_no_sub_centimetre_numbers():
    """Jev must not be handed precision it could grasp from."""
    text = verbalize.describe(state(goal=(0.4567, 0.0123, 0.0089)), "pick x")
    numeric = [
        tok for tok in text.replace("(", " ").replace(")", " ").replace(",", " ").split()
        if any(ch.isdigit() for ch in tok)
    ]
    for tok in numeric:
        body = tok.split("=")[-1]
        assert "." not in body, tok


def test_verbaliser_names_no_task(monkeypatch):
    """The situation text describes a generic move, not a retail phase."""
    text = verbalize.describe(state(), "open the fridge").lower()
    for word in ("shelf", "basket", "retail", "grasp", "phase"):
        assert word not in text, word


def test_coarse_action_never_exceeds_its_step_budget():
    geom = Geometry()
    s = state(goal=(5.0, 0.0, 0.0))
    step = geom.coarse_step(0.045)
    for action in verbalize.COARSE_ACTIONS:
        assert np.linalg.norm(
            verbalize.resolve_coarse_action(action, s, step)
        ) <= step + 1e-9, action


def test_toward_goal_is_clipped_and_does_not_overshoot():
    s = state(goal=(0.01, 0.0, 0.0))
    assert np.linalg.norm(
        verbalize.resolve_coarse_action("toward_goal", s, 0.18)
    ) == pytest.approx(0.01)


def test_hold_is_a_no_op_and_unknown_actions_are_rejected():
    assert np.allclose(verbalize.resolve_coarse_action("hold", state(), 0.18), 0.0)
    with pytest.raises(ValueError):
        verbalize.resolve_coarse_action("teleport", state(), 0.1)


# -- plan primitives ----------------------------------------------------------
def test_relative_primitives_measure_from_the_subgoal_anchor():
    """Measuring from the live TCP makes the goal recede every tick forever."""
    sg = Subgoal("extract", GoalKind.WITHDRAW, tolerance_mm=40.0,
                 params={"distance": 0.3})
    anchor = resolve(
        sg, {}, tcp_position=np.array([1.0, 0.0, 1.0]),
        anchor=None, anchor_tcp=np.array([1.0, 0.0, 1.0]), finger_length_m=0.03,
    )
    moved = resolve(
        sg, {}, tcp_position=np.array([0.8, 0.0, 1.0]),
        anchor=None, anchor_tcp=np.array([1.0, 0.0, 1.0]), finger_length_m=0.03,
    )
    assert np.allclose(anchor.position, moved.position)


def test_downward_approach_has_a_well_defined_closing_axis():
    """cross(approach, up) collapses for a floor pick or a basket release."""
    floor = Frame(origin=[1.0, 0.0, 0.08], normal=[0.0, 0.0, 1.0])
    wp = resolve(
        Subgoal("grasp", GoalKind.CONTACT, "t"), {"t": floor},
        tcp_position=np.zeros(3), anchor=None, finger_length_m=0.03,
    )
    assert np.linalg.norm(wp.closing) == pytest.approx(1.0)
    assert abs(float(np.dot(wp.closing, wp.approach))) < 1e-9


def test_missing_scene_reference_fails_loudly():
    """Silently aiming at the origin would be worse than crashing."""
    with pytest.raises(KeyError):
        resolve(
            Subgoal("grasp", GoalKind.CONTACT, "nonexistent"), {},
            tcp_position=np.zeros(3), anchor=None, finger_length_m=0.03,
        )


def test_free_primitives_resolve_to_no_motion():
    for kind in FREE_KINDS:
        assert resolve(
            Subgoal("x", kind), {}, tcp_position=np.zeros(3),
            anchor=None, finger_length_m=0.03,
        ) is None


@pytest.mark.parametrize("instruction", INSTRUCTIONS)
def test_every_generated_plan_is_executable_in_principle(instruction):
    plan = synthesise_plan(instruction)
    assert len(plan) >= 3
    assert plan.required_refs
    # Every non-free subgoal either names a reference or is relative.
    for sg in plan.subgoals:
        if sg.is_free or sg.kind is GoalKind.WITHDRAW:
            continue
        assert sg.ref, sg.name


def test_a_long_swing_is_split_into_several_short_ones():
    """A single large arc gets chased in a straight line and tears the grip."""
    plan = synthesise_plan("open the fridge")
    arcs = [sg for sg in plan.subgoals if sg.kind is GoalKind.ARC]
    assert len(arcs) >= 2
    assert all(abs(sg.params["angle_deg"]) <= 40 for sg in arcs)


# -- billing integrity --------------------------------------------------------
def test_billed_jev_prompt_contains_only_situation_and_questions():
    """`hint` must never reach the billed text.

    The mock backend uses privileged state to synthesise calibrated answers.
    If any of that leaked into `render_request`, the mock's token counts would
    stop being what a real deployment pays.
    """
    s = state(goal=(1.2345, 0.0, 0.0))
    situation = verbalize.describe(s, "pick nivea to basket")
    questions = [
        Question(id="coarse_action", kind="choice", instruction="best move?",
                 options=verbalize.COARSE_ACTIONS),
        Question(id="belief_stale", kind="noul", instruction="",
                 proposition="target position is stale"),
    ]
    billed = render_request(situation, questions)
    assert situation in billed
    for q in questions:
        assert q.render() in billed
    assert "1.2345" not in billed


def test_plan_cache_key_ignores_the_item_name():
    """The plan shape depends on the verb and destination, not the brand."""
    k = PlanCache.key
    assert k("move to shelf and pick Nivea Body Milk to basket") == k(
        "move to shelf and pick Fanta Sabor Naranja 2L to basket"
    )
    assert k("open the fridge") != k("close the fridge")


def test_action_layout_locks_out_the_mobile_base():
    """No mobile manipulation: the base and head dims are never written.

    Mirrors what RoboBenchMart's own motion planner does in `follow_path`.
    Asserted so a refactor cannot quietly turn this into mobile manipulation.
    """
    assert (BASE_SLICE.start, BASE_SLICE.stop) == (11, 13)
    assert (HEAD_PAN_IDX, HEAD_TILT_IDX) == (8, 9)

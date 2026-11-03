"""Tests for the experiment's integrity conditions.

These are not unit tests of convenience. Each one guards a property that, if
it broke silently, would invalidate the cost comparison rather than merely
produce a bug.
"""

import numpy as np
import pytest

from astraxjev import verbalize
from astraxjev.backends.base import Question, render_request
from astraxjev.pipelines import Geometry
from astraxjev.pipelines.astra_jev import AstraJevPipeline
from astraxjev.types import (
    BASE_SLICE,
    FINE_PRECISION_MM,
    HEAD_PAN_IDX,
    HEAD_TILT_IDX,
    PHASE_SPEC,
    Phase,
    SymbolicState,
)


def hybrid():
    return AstraJevPipeline(astra=None, jev=None, geom=Geometry())


# -- who is allowed to do what ------------------------------------------------
def test_jev_never_owns_a_fine_phase():
    """The whole premise: millimetre work stays with Astra.

    If this ever passes for ALIGN or GRASP, the hybrid is no longer the
    experiment described -- it is a cheaper, less precise policy, and any cost
    win would be bought with precision rather than with routing.
    """
    h = hybrid()
    for phase in (Phase.ALIGN, Phase.GRASP):
        assert not h.jev_owns(phase), phase


def test_jev_owns_exactly_the_coarse_phases():
    h = hybrid()
    assert {p for p in Phase if h.jev_owns(p)} == {
        Phase.APPROACH,
        Phase.RETRACT,
        Phase.PLACE,
        Phase.SETTLE,
        Phase.DONE,
    }


def test_locate_needs_pixels_so_it_cannot_be_delegated():
    """Only an MLLM maps an instruction plus a cluttered shelf to a 3D point."""
    assert PHASE_SPEC[Phase.LOCATE]["needs_pixels"]
    assert not hybrid().jev_owns(Phase.LOCATE)


def test_phase_assignment_is_derived_not_hardcoded():
    """Raising the threshold must hand more phases to Jev, and lowering it fewer.

    Phase assignment follows from the declared tolerances, so changing the
    policy is a config change, not a code change.
    """
    permissive = AstraJevPipeline(None, None, Geometry(), jev_precision_threshold_mm=1.0)
    strict = AstraJevPipeline(None, None, Geometry(), jev_precision_threshold_mm=1e6)
    assert permissive.jev_owns(Phase.RETRACT)
    assert not strict.jev_owns(Phase.PLACE)
    # Even at a permissive threshold, pixel-bound phases stay with Astra.
    assert not permissive.jev_owns(Phase.ALIGN)


# -- the coarse interface cannot fake fine precision --------------------------
def state(phase=Phase.APPROACH, tcp=(0.0, 0.0, 0.0), standoff=(0.5, 0.0, 0.0)):
    return SymbolicState(
        phase=phase,
        tcp_position=np.array(tcp, dtype=float),
        target_position=np.array([0.7, 0.0, 0.0]),
        standoff_position=np.array(standoff, dtype=float),
        basket_position=np.array([0.0, 0.3, -0.3]),
        shelf_normal=np.array([-1.0, 0.0, 0.0]),
        gripper_closed=False,
        holding_item=False,
        nearest_neighbour_cm=20.0,
        steps_in_phase=1,
        steps_total=1,
    )


def test_coarse_step_is_an_order_of_magnitude_above_fine_tolerance():
    geom = Geometry()
    fine_m = FINE_PRECISION_MM / 1000.0
    for phase in (Phase.APPROACH, Phase.RETRACT, Phase.PLACE):
        tol = PHASE_SPEC[phase]["precision_mm"] / 1000.0
        assert geom.coarse_step(tol) >= 10 * fine_m, phase


def test_verbaliser_emits_only_whole_centimetres():
    """Jev must not be handed sub-centimetre numbers it could grasp from."""
    text = verbalize.describe(state(), "pick nivea to basket")
    for token in text.replace("(", " ").replace(")", " ").replace(",", " ").split():
        if token.lstrip("-").isdigit():
            continue
        # No bare decimals anywhere in the numeric fields.
        assert "." not in token or "=" in token or "/" in token, token


def test_coarse_action_never_exceeds_its_step_budget():
    geom = Geometry()
    s = state(tcp=(0.0, 0.0, 0.0), standoff=(5.0, 0.0, 0.0))  # goal far away
    step = geom.coarse_step(PHASE_SPEC[Phase.APPROACH]["precision_mm"] / 1000.0)
    for action in verbalize.COARSE_ACTIONS:
        delta = verbalize.resolve_coarse_action(action, s, step)
        assert np.linalg.norm(delta) <= step + 1e-9, action


def test_toward_goal_is_clipped_and_does_not_overshoot():
    s = state(tcp=(0.0, 0.0, 0.0), standoff=(0.01, 0.0, 0.0))
    delta = verbalize.resolve_coarse_action("toward_goal", s, 0.16)
    assert np.linalg.norm(delta) == pytest.approx(0.01)


def test_hold_is_a_no_op():
    assert np.allclose(verbalize.resolve_coarse_action("hold", state(), 0.16), 0.0)


def test_unknown_coarse_action_is_rejected():
    with pytest.raises(ValueError):
        verbalize.resolve_coarse_action("teleport_to_goal", state(), 0.1)


# -- billing integrity --------------------------------------------------------
def test_billed_jev_prompt_contains_only_situation_and_questions():
    """`hint` must never reach the billed text.

    The mock backend uses privileged state to synthesise calibrated answers. If
    any of that leaked into `render_request`, the mock's token counts would no
    longer be what a real deployment pays, and the whole cost measurement would
    stop transferring.
    """
    s = state()
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
    # The true target position is privileged; it must not appear.
    assert "0.7" not in billed


def test_action_layout_locks_out_the_mobile_base():
    """No mobile manipulation: the base and head dims are never written.

    This mirrors what RoboBenchMart's own motion planner does in
    `follow_path`, which writes zeros into the last two action dims and holds
    the body joints. The constants are asserted here so a refactor cannot
    quietly turn this into a mobile-manipulation task.
    """
    assert (BASE_SLICE.start, BASE_SLICE.stop) == (11, 13)
    assert (HEAD_PAN_IDX, HEAD_TILT_IDX) == (8, 9)

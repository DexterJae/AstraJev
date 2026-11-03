"""Free perception.

Both pipelines get these at no charge, and that is deliberate: forward
kinematics from joint encoders, gripper width, and the pose of a basket
rigidly bolted to the robot are all free on a real robot. Charging a model
call for them would make the comparison meaningless in either direction.

Everything that is *not* free -- which item on the shelf is the target, where
it is in 3D, how close the gripper is to a neighbour -- comes from a model
call and is recorded in the cost ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .oracle import Oracle
from .plan import Frame
from .types import Observation


@dataclass
class Senses:
    tcp_position: np.ndarray
    gripper_closed: bool
    holding_item: bool
    #: Scene references obtainable without a model call, keyed by plan ref
    #: name. A basket bolted to the base is the canonical example.
    free_frames: dict[str, Frame] = field(default_factory=dict)


class ProprioSenses:
    """Proprioceptive readout.

    On a real robot this is forward kinematics over `obs.state` plus the
    robot's own URDF. Under the mock backends it is read from the oracle,
    which is the same number FK would produce; the oracle is used only to
    avoid shipping a URDF FK dependency into the cost-accounting tests.
    """

    def __init__(self, oracle: Oracle | None = None) -> None:
        self.oracle = oracle

    def read(self, obs: Observation) -> Senses:
        if self.oracle is None:
            raise RuntimeError(
                "ProprioSenses needs either an oracle (sim) or a URDF FK "
                "implementation; see the docs in this module."
            )
        return Senses(
            tcp_position=self.oracle.tcp_position(),
            gripper_closed=self.oracle.gripper_closed(),
            holding_item=self.oracle.holding_item(),
            free_frames=self.free_frames(obs),
        )

    def free_frames(self, obs: Observation) -> dict[str, Frame]:
        """Frames that cost nothing because they are rigidly known."""
        if self.oracle is None:
            return {}
        return {
            name: f
            for name, f in self.oracle.frames().items()
            if name in self.oracle.free_ref_names()
        }

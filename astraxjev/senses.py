"""Free proprioception.

Both pipelines get these quantities at no charge, and that is deliberate:
forward kinematics from joint encoders, gripper width, and the pose of a basket
rigidly bolted to the robot are all free on a real robot. Charging a model call
for them would make the comparison meaningless in either direction.

Everything that is *not* free -- which item on the shelf is the target, where it
is in 3D, how close the gripper is to a neighbouring item -- comes from a model
call and is recorded in the cost ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracle import Oracle
from .types import Observation


@dataclass
class Senses:
    tcp_position: np.ndarray
    basket_position: np.ndarray
    gripper_closed: bool
    holding_item: bool


class ProprioSenses:
    """Proprioceptive readout.

    On a real robot this is forward kinematics over `obs.state`. Under the mock
    backends it is read from the oracle, which is the same number FK would
    produce; the oracle is used only to avoid shipping a URDF FK dependency
    into the cost-accounting tests.
    """

    def __init__(self, oracle: Oracle | None = None) -> None:
        self.oracle = oracle

    def read(self, obs: Observation) -> Senses:
        if self.oracle is None:
            raise RuntimeError(
                "ProprioSenses needs either an oracle (sim) or a URDF FK "
                "implementation; see docs in this module."
            )
        return Senses(
            tcp_position=self.oracle.tcp_position(),
            basket_position=self.oracle.basket_position(),
            gripper_closed=self.oracle.gripper_closed(),
            holding_item=self.oracle.holding_item(),
        )

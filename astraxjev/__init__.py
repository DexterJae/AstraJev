"""Astra vs Astra+Jev on a RoboBenchMart gripper task.

Two pipelines solve the same PickToBasket episode with the mobile base locked,
through the same phase machine and the same low-level executor, so that the
only difference between them is which model makes each decision and how often.
"""

__all__ = ["cost", "geometry", "pipelines", "prompts", "tokens", "types", "verbalize"]

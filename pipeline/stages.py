from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Stage:
    name: str
    runner: Callable
    depends_on: tuple[str, ...] = ()


class StageRegistry:
    def __init__(self, stages):
        stages = list(stages)
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("duplicate stage name")
        if any("/" in name or "\\" in name for name in names):
            raise ValueError("stage name cannot contain a path separator")

        known = set(names)
        for stage in stages:
            unknown = set(stage.depends_on) - known
            if unknown:
                raise ValueError(f"unknown stage dependency: {sorted(unknown)[0]}")

        remaining = list(stages)
        ordered = []
        completed = set()
        while remaining:
            ready = next(
                (stage for stage in remaining if set(stage.depends_on) <= completed),
                None,
            )
            if ready is None:
                raise ValueError("stage dependency cycle")
            remaining.remove(ready)
            ordered.append(ready)
            completed.add(ready.name)

        self._ordered = tuple(ordered)
        self._snapshot = {
            "stage_order": [stage.name for stage in ordered],
            "stage_directories": {
                stage.name: f"{number:02d}_{stage.name}"
                for number, stage in enumerate(ordered, 1)
            },
            "stage_dependencies": {
                stage.name: list(stage.depends_on) for stage in ordered
            },
        }

    def ordered(self):
        return list(self._ordered)

    def snapshot(self):
        return deepcopy(self._snapshot)

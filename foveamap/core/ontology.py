"""Authoritative semantic class definitions and ontology for FoveaMap.

Defines the single source of truth for semantic class IDs, names, grouping,
and validation across perception, datasets, and mapping layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


CANONICAL_CLASSES: tuple[str, ...] = (
    "road",
    "sidewalk",
    "parking",
    "terrain",
    "vegetation",
    "building",
    "pole",
    "vehicle",
    "person",
)

ROAD, SIDEWALK, PARKING, TERRAIN, VEGETATION, BUILDING, POLE, VEHICLE, PERSON = range(9)
NUM_CLASSES: int = len(CANONICAL_CLASSES)

GROUND_CLASSES: tuple[int, ...] = (ROAD, SIDEWALK, PARKING, TERRAIN)
DRIVABLE_CLASSES: tuple[int, ...] = (ROAD, PARKING)
DYNAMIC_CLASSES: tuple[int, ...] = (VEHICLE, PERSON)


@dataclass(frozen=True)
class SemanticOntology:
    """Authoritative semantic class ontology configuration and helpers."""
    class_names: tuple[str, ...] = CANONICAL_CLASSES
    num_classes: int = NUM_CLASSES
    active_classes: tuple[bool, ...] = field(default_factory=lambda: (True,) * NUM_CLASSES)

    def __post_init__(self) -> None:
        if self.num_classes != len(self.class_names):
            raise ValueError(
                f"num_classes ({self.num_classes}) must match class_names length ({len(self.class_names)})"
            )
        if len(self.active_classes) != self.num_classes:
            raise ValueError(
                f"active_classes length ({len(self.active_classes)}) must match num_classes ({self.num_classes})"
            )

    def name_of(self, class_id: int) -> str:
        """Return human-readable class name for a given class ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        raise IndexError(f"Class ID {class_id} out of bounds for ontology with {self.num_classes} classes")

    def id_of(self, class_name: str) -> int:
        """Return class ID for a given class name."""
        try:
            return self.class_names.index(class_name.lower().strip())
        except ValueError:
            raise KeyError(f"Unknown class name {class_name!r}; valid classes: {self.class_names}")

    def is_ground(self, class_id: int) -> bool:
        """Check if class ID belongs to ground/surface category."""
        return class_id in GROUND_CLASSES

    def is_drivable(self, class_id: int) -> bool:
        """Check if class ID represents drivable road surface."""
        return class_id in DRIVABLE_CLASSES

    def is_dynamic(self, class_id: int) -> bool:
        """Check if class ID represents inherently dynamic objects (vehicle, person)."""
        return class_id in DYNAMIC_CLASSES

    def validate_class_id(self, class_id: int) -> bool:
        """Check whether class_id is within the valid ontology range."""
        return 0 <= class_id < self.num_classes


DEFAULT_ONTOLOGY = SemanticOntology()

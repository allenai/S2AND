"""Deterministic pair scores for clustering policy tests."""

import numpy as np


class ConstantDistanceClassifier:
    """Expose a fixed non-match probability without fitting an unrelated model."""

    def __init__(self, distance: float = 0.3):
        self.distance = distance

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return two-class probabilities for each requested pair."""
        return np.tile([self.distance, 1.0 - self.distance], (len(features), 1))

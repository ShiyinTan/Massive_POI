from __future__ import annotations

from typing import Dict

import numpy as np

from ..information.calibration import summarize_calibration


def evaluate_calibration(probs_by_subset: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    return summarize_calibration(probs_by_subset, labels)
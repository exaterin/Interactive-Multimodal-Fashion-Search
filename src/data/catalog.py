from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class Catalog:
    image_ids: List[str]
    embeddings: np.ndarray
    image_paths: Dict[str, Path]

    fabric_annotations: Dict[str, Dict[str, str]]
    pattern_annotations: Dict[str, Dict[str, str]]
    shape_annotations: Dict[str, Dict[str, str]]

    available_values: Dict[str, List[str]]
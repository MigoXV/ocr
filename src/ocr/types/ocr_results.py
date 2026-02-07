from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class OCRResult:
    ref: str
    det: np.ndarray


OCRResults = List[OCRResult]

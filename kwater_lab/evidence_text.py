"""Deterministic Korean evidence vectors shared by fold-local training experiments."""

from functools import lru_cache
import re
import zlib

import numpy as np


REMOVED = re.compile(
    r"[A-Za-z]+(?:_[A-Za-z]+)+|제\s*\d+\s*조(?:의\s*\d+)?|"
    r"추락|낙상|전도|걸림|끼임|협착|충돌|낙하|비래|감전|익수|침수|화재|폭발|밀폐공간|열사병|한랭질환|위험성?|재해"
)


@lru_cache(maxsize=1024)
def text_vector(observation):
    clean = re.sub(
        r"[^가-힣a-z0-9]+", " ", REMOVED.sub(" ", observation.lower())
    ).strip()
    vector = np.zeros(512)
    for n in (2, 3):
        for i in range(len(clean) - n + 1):
            bucket = zlib.crc32(clean[i : i + n].encode())
            vector[bucket % len(vector)] += 1 if (bucket >> 31) == 0 else -1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector

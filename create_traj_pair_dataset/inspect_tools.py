from typing import Dict, List, Tuple
import random
import numpy as np


def select_inspection_pairs(pos_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                            neg_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                            k: int,
                            rng: random.Random):
    # Mix of positives and negatives, stratified by length buckets (approximate)
    sel = []
    def take(src):
        if not src:
            return []
        idxs = list(range(len(src)))
        rng.shuffle(idxs)
        return [src[i] for i in idxs[:max(0, k // 2)]]
    sel.extend(take(pos_pairs))
    sel.extend(take(neg_pairs))
    rng.shuffle(sel)
    return sel[:k]

__all__ = ["select_inspection_pairs"]

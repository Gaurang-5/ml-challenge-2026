"""High-recall in-memory multi-key inverted index for entity blocking.

Memory-optimized using array('I') integer posting lists.
"""
from __future__ import annotations

import array
from collections import defaultdict
from normalizer import clean_business_name, clean_address


def generate_blocking_keys(name: str, addr: str) -> set[str]:
    """Generates redundant, high-specificity and high-recall keys for an entity."""
    norm_name, name_tokens = clean_business_name(name)
    norm_addr, addr_tokens, numbers = clean_address(addr)
    
    keys: set[str] = set()
    
    # 1. Full normalized name
    if norm_name:
        keys.add(f"N_FULL|{norm_name}")
        
    # 2. Distinctive name tokens & prefixes
    for tok in name_tokens:
        if len(tok) >= 3:
            keys.add(f"N_TOK|{tok}")
        if len(tok) >= 4:
            keys.add(f"N_PRE|{tok[:4]}")
            
    # 3. Name token 2-grams (captures multi-word company identity)
    for i in range(len(name_tokens) - 1):
        keys.add(f"N_BI|{name_tokens[i]}_{name_tokens[i+1]}")
        
    # 4. Numbers in address (e.g. house number, PIN code, postal code, suite)
    for num in numbers:
        if len(num) >= 2:
            keys.add(f"NUM|{num}")
            
    # 5. Distinctive address tokens and 2-grams
    for tok in addr_tokens:
        if len(tok) >= 3 and not tok.isdigit():
            keys.add(f"A_TOK|{tok}")
    for i in range(len(addr_tokens) - 1):
        if not addr_tokens[i].isdigit() and not addr_tokens[i+1].isdigit():
            keys.add(f"A_BI|{addr_tokens[i]}_{addr_tokens[i+1]}")
            
    # 6. Combined street number + street token (strong spatial anchor)
    for num in numbers[:2]:
        for a_tok in addr_tokens[:2]:
            keys.add(f"NUM_A|{num}_{a_tok}")
            
    return keys


class InvertedIndex:
    """In-memory inverted index of blocking keys to target integer IDs."""
    
    def __init__(self) -> None:
        self.postings: dict[str, array.array] = {}
        
    def add(self, target_idx: int, keys: set[str]) -> None:
        for key in keys:
            if key not in self.postings:
                self.postings[key] = array.array("I")
            self.postings[key].append(target_idx)
            
    def get_candidates(
        self,
        query_keys: set[str],
        max_key_frequency: int = 1200,
        top_k: int = 60,
    ) -> list[tuple[int, int]]:
        """Scores candidate targets by key specificity and returns top_k (target_idx, score)."""
        scores: dict[int, int] = defaultdict(int)
        
        for k in query_keys:
            post_list = self.postings.get(k)
            if post_list is None:
                continue
            count = len(post_list)
            if count > max_key_frequency:
                continue
                
            # Key specificity weighting
            if "N_FULL|" in k or "NUM_A|" in k:
                w = 8
            elif "N_BI|" in k or "A_BI|" in k:
                w = 5
            elif "NUM|" in k and len(k.split("|")[-1]) >= 4:
                w = 5
            elif "N_TOK|" in k:
                w = 3
            elif "NUM|" in k:
                w = 2
            else:
                w = 1
                
            for tid in post_list:
                scores[tid] += w
                
        if not scores:
            return []
            
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return ranked

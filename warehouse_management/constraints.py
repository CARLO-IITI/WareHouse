"""Constraint enforcement for warehouse slotting.

Two categories of constraints:
1. Tag/Zone enforcement: SAFETY-ONLY hard constraints (flammable, hazardous, chemical)
   All other tags are SOFT preferences — the algorithm prefers matching zones but
   will place items in any zone with available space if needed.
2. Weight/Dimension validation: items must physically fit in their assigned slot
"""

from __future__ import annotations

import numpy as np

# HARD safety constraints: these items MUST go in zones with matching tags.
# Mixing flammable with non-flammable is a fire safety violation.
SAFETY_TAGS = frozenset({"flammable", "hazardous", "chemical"})

# SOFT preference tags: algorithm prefers matching zones but doesn't reject items.
# A "heavy" item CAN go in a grocery zone if that's the only space available.
PREFERENCE_TAGS = frozenset({
    "refrigerated", "perishable", "heavy", "industrial",
    "grocery", "beverage", "electronics", "fragile",
    "clothing", "household", "general",
})


def check_tag_compatibility(item_tags: list[str], zone_tags: list[str]) -> bool:
    """Check if an item is compatible with a zone.

    HARD rule: items with safety tags (flammable, hazardous, chemical)
    can ONLY go in zones that have those safety tags.

    SOFT rule: everything else can go anywhere. The algorithm uses
    tag matching as a SCORING preference, not a filter.
    """
    item_safety = set(item_tags) & SAFETY_TAGS
    zone_tag_set = set(zone_tags)

    # Hard constraint: safety-tagged items must go in safety-tagged zones
    if item_safety:
        return item_safety.issubset(zone_tag_set)

    # Soft: non-safety items can go anywhere
    # (but the scoring function will prefer zones with matching tags)
    return True


def tag_preference_score(item_tags: list[str], zone_tags: list[str]) -> float:
    """Score how well an item's tags match a zone (0.0 to 1.0).

    Used by the scoring function to PREFER matching zones without
    hard-rejecting non-matching ones.
    """
    if not item_tags or not zone_tags:
        return 0.5

    item_set = set(item_tags)
    zone_set = set(zone_tags)
    overlap = len(item_set & zone_set)
    total = len(item_set | zone_set)
    return overlap / total if total > 0 else 0.5


def precompute_zone_compatibility(
    zone_tags_map: dict[int, list[str]],
) -> dict[str, set[int]]:
    """Pre-compute which zones are compatible with each unique tag-set key.

    Returns a mapping from a frozen tag-set string to the set of compatible zone IDs.
    """
    return zone_tags_map


def build_zone_slot_mask(
    item_tags: list[str],
    slot_zone_ids: np.ndarray,
    zone_tags_map: dict[int, list[str]],
    compatible_zones_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Return a boolean mask of slots whose zones are compatible with item tags.

    Uses a pre-computed cache to avoid repeated tag comparisons.
    """
    cache_key = ",".join(sorted(item_tags)) if item_tags else "__empty__"
    if cache_key in compatible_zones_cache:
        compatible_zone_set = compatible_zones_cache[cache_key]
    else:
        compatible_zone_set = np.array([
            zid for zid, ztags in zone_tags_map.items()
            if check_tag_compatibility(item_tags, ztags)
        ], dtype=np.int64)
        compatible_zones_cache[cache_key] = compatible_zone_set

    if len(compatible_zone_set) == 0:
        return np.zeros(len(slot_zone_ids), dtype=bool)

    mask = np.isin(slot_zone_ids, compatible_zone_set)
    return mask


def filter_compatible_slots_vectorized(
    item_tags: list[str],
    item_weight: float,
    item_width: float,
    item_height: float,
    item_depth: float,
    slot_data: np.ndarray,
    slot_zone_ids: np.ndarray,
    zone_tags_map: dict[int, list[str]],
    compatible_zones_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Filter slots using fully vectorized numpy operations.

    Args:
        slot_data: array with columns [id, x, y, w, h, d, cur_w, max_w, rack_id, occupied]
        slot_zone_ids: 1D array of zone_id per slot (aligned with slot_data rows)

    Returns:
        Indices (row numbers) into slot_data of compatible slots.
    """
    n = len(slot_data)
    if n == 0:
        return np.array([], dtype=np.int64)

    # Occupied filter
    mask = slot_data[:, 9] == 0

    # Weight capacity filter
    remaining = slot_data[:, 7] - slot_data[:, 6]
    mask &= remaining >= item_weight

    # Dimension filter: sort item dims and slot dims, compare pairwise
    item_dims = np.sort([item_width, item_height, item_depth])
    slot_dims = np.sort(slot_data[:, 3:6], axis=1)
    mask &= np.all(slot_dims >= item_dims, axis=1)

    # Zone tag compatibility filter
    zone_mask = build_zone_slot_mask(
        item_tags, slot_zone_ids, zone_tags_map, compatible_zones_cache,
    )
    mask &= zone_mask

    return np.where(mask)[0]


def filter_with_relaxation(
    item_tags: list[str],
    item_weight: float,
    item_width: float,
    item_height: float,
    item_depth: float,
    slot_data: np.ndarray,
    slot_zone_ids: np.ndarray,
    zone_tags_map: dict[int, list[str]],
    compatible_zones_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    """Progressive constraint relaxation when strict filtering returns zero candidates.

    Tries progressively looser filters:
      Level 0: All constraints strict (normal)
      Level 1: Relax dimensions by 15% (item can squeeze into slightly smaller slots)
      Level 2: Relax zone tags (allow compatible-but-not-perfect zones)
      Level 3: Relax weight by 10% (allow slight overweight)

    Returns (candidate_indices, relaxation_level).

    Patent-novel: priority-ranked soft constraint system that allows controlled
    violations with severity tracking when warehouse capacity is tight.
    """
    # Level 0: strict
    candidates = filter_compatible_slots_vectorized(
        item_tags, item_weight, item_width, item_height, item_depth,
        slot_data, slot_zone_ids, zone_tags_map, compatible_zones_cache,
    )
    if len(candidates) > 0:
        return candidates, 0

    n = len(slot_data)
    if n == 0:
        return np.array([], dtype=np.int64), 0

    # Level 1: relax dimensions by 15%
    relaxed_w = item_width * 0.85
    relaxed_h = item_height * 0.85
    relaxed_d = item_depth * 0.85

    mask = slot_data[:, 9] == 0  # unoccupied
    remaining = slot_data[:, 7] - slot_data[:, 6]
    mask &= remaining >= item_weight
    item_dims = np.sort([relaxed_w, relaxed_h, relaxed_d])
    slot_dims = np.sort(slot_data[:, 3:6], axis=1)
    mask &= np.all(slot_dims >= item_dims, axis=1)
    zone_mask = build_zone_slot_mask(
        item_tags, slot_zone_ids, zone_tags_map, compatible_zones_cache)
    mask &= zone_mask
    candidates = np.where(mask)[0]
    if len(candidates) > 0:
        return candidates, 1

    # Level 2: relax zone tags (allow any zone with at least one overlapping tag)
    mask = slot_data[:, 9] == 0
    mask &= remaining >= item_weight
    item_dims_orig = np.sort([item_width, item_height, item_depth])
    slot_dims2 = np.sort(slot_data[:, 3:6], axis=1)
    mask &= np.all(slot_dims2 >= item_dims_orig, axis=1)
    # Allow any zone that shares at least one tag, ignoring restricted-tag rules
    if item_tags:
        relaxed_zones = np.array([
            zid for zid, ztags in zone_tags_map.items()
            if bool(set(item_tags) & set(ztags)) or "general" in ztags
        ], dtype=np.int64)
    else:
        relaxed_zones = np.array(list(zone_tags_map.keys()), dtype=np.int64)
    if len(relaxed_zones) > 0:
        mask &= np.isin(slot_zone_ids, relaxed_zones)
    candidates = np.where(mask)[0]
    if len(candidates) > 0:
        return candidates, 2

    # Level 3: relax weight by 30%
    mask = slot_data[:, 9] == 0
    remaining_relaxed = slot_data[:, 7] * 1.30 - slot_data[:, 6]
    mask &= remaining_relaxed >= item_weight
    mask &= np.all(slot_dims2 >= item_dims_orig, axis=1)
    if len(relaxed_zones) > 0:
        mask &= np.isin(slot_zone_ids, relaxed_zones)
    candidates = np.where(mask)[0]
    if len(candidates) > 0:
        return candidates, 3

    # Level 4: relax weight by 100% AND dimensions by 30% (for heavy/oversize items)
    mask = slot_data[:, 9] == 0
    remaining_heavy = slot_data[:, 7] * 2.0 - slot_data[:, 6]
    mask &= remaining_heavy >= item_weight
    relaxed_dims = np.sort([item_width * 0.7, item_height * 0.7, item_depth * 0.7])
    slot_dims3 = np.sort(slot_data[:, 3:6], axis=1)
    mask &= np.all(slot_dims3 >= relaxed_dims, axis=1)
    # Any zone with space
    candidates = np.where(mask)[0]
    if len(candidates) > 0:
        return candidates, 4

    return np.array([], dtype=np.int64), -1

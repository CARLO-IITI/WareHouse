"""Core slotting engine: DBSCAN + Genetic Algorithm hybrid.

Phase 1: DBSCAN clusters items by co-purchase affinity (arbitrary-shaped clusters)
Phase 2: Genetic Algorithm optimizes cluster-to-zone assignment (global optimization)
Phase 3: Within each zone, items sorted by velocity and assigned to slots with
          ergonomic golden-zone scoring and constraint relaxation.
"""

from __future__ import annotations

import json
import random as py_random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text
from sqlalchemy.orm import Session
try:
    import sys; sys.stderr.flush()
    from tqdm import tqdm
except (BrokenPipeError, OSError):
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it) if self._it else iter([])
        def update(self, n=1): pass
        def close(self): pass

from .constraints import check_tag_compatibility, tag_preference_score, filter_compatible_slots_vectorized, filter_with_relaxation
from .warehouse_graph import WarehouseGraph, build_warehouse_graph, slot_to_picking_distance


# ---------------------------------------------------------------------------
# Velocity Scoring (ABC Analysis) — unchanged
# ---------------------------------------------------------------------------

def compute_velocity_scores(session: Session) -> dict[int, tuple[float, str]]:
    """Compute velocity scores for all items using order frequency."""
    print("  Computing velocity scores from order history...")
    t0 = time.time()

    result = session.execute(text("""
        SELECT item_id, COUNT(DISTINCT order_id) as order_count
        FROM order_history GROUP BY item_id ORDER BY order_count DESC
    """))
    rows = result.fetchall()
    if not rows:
        return {}

    item_order_counts = {row[0]: row[1] for row in rows}
    max_count = max(item_order_counts.values()) if item_order_counts else 1
    scores = {iid: count / max_count for iid, count in item_order_counts.items()}
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    cutoff_a, cutoff_b = int(n * 0.20), int(n * 0.50)

    result_map = {}
    for i, (item_id, score) in enumerate(sorted_items):
        cls = "A" if i < cutoff_a else ("B" if i < cutoff_b else "C")
        result_map[item_id] = (score, cls)

    for r in session.execute(text("SELECT id FROM items")).fetchall():
        if r[0] not in result_map:
            result_map[r[0]] = (0.0, "C")

    elapsed = time.time() - t0
    a = sum(1 for v in result_map.values() if v[1] == "A")
    b = sum(1 for v in result_map.values() if v[1] == "B")
    c = sum(1 for v in result_map.values() if v[1] == "C")
    print(f"  Velocity scores computed for {len(result_map)} items in {elapsed:.1f}s")
    print(f"    A-class (top 20%): {a} | B-class (30%): {b} | C-class (50%): {c}")
    return result_map


def update_velocity_scores_in_db(session: Session, velocity_map: dict[int, tuple[float, str]]) -> None:
    """Batch-update velocity scores and classes in the items table."""
    print("  Persisting velocity scores to database...")
    items = list(velocity_map.items())
    for start in range(0, len(items), 5000):
        for item_id, (score, cls) in items[start:start + 5000]:
            session.execute(text("UPDATE items SET velocity_score=:s, velocity_class=:c WHERE id=:id"),
                            {"s": score, "c": cls, "id": item_id})
        session.flush()
    session.commit()
    print("  Velocity scores persisted.")


# ---------------------------------------------------------------------------
# Phase 1: DBSCAN Affinity Clustering (replaces BallTree KNN)
# ---------------------------------------------------------------------------

@dataclass
class AffinityIndex:
    """DBSCAN-based affinity clusters.

    Keeps the same class name for backward compatibility with dashboard imports.
    Instead of KNN neighbors, stores cluster labels and membership dicts.
    """
    item_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    item_id_to_idx: dict[int, int] = field(default_factory=dict)
    cluster_labels: np.ndarray = None
    clusters: dict[int, list[int]] = field(default_factory=dict)
    n_clusters: int = 0
    noise_items: list[int] = field(default_factory=list)
    reduced_vectors: np.ndarray = None
    # Keep nn_model for backward compat with queries that use get_affinity_neighbors
    nn_model: NearestNeighbors | None = None
    k_neighbors: int = 10


def _auto_tune_dbscan_eps(vectors: np.ndarray, k: int = 5) -> float:
    """Auto-tune DBSCAN eps using the k-distance graph elbow method.

    Computes the k-th nearest neighbor distance for all points,
    sorts them, and picks the "knee" point where distance starts growing fast.
    """
    if len(vectors) < k + 1:
        return 1.0

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(vectors)
    distances, _ = nn.kneighbors(vectors)
    k_dists = np.sort(distances[:, k])

    # Find the elbow: the point of maximum curvature
    # Simple approach: find where the second derivative is maximum
    if len(k_dists) < 10:
        return float(np.median(k_dists))

    # Compute numerical second derivative
    d1 = np.diff(k_dists)
    d2 = np.diff(d1)
    if len(d2) == 0:
        return float(np.median(k_dists))

    elbow_idx = np.argmax(d2) + 1
    eps = float(k_dists[min(elbow_idx, len(k_dists) - 1)])

    # Sanity bounds
    eps = max(eps, 0.1)
    eps = min(eps, float(np.percentile(k_dists, 95)))

    return eps


def build_affinity_index(session: Session, k_neighbors: int = 10) -> AffinityIndex:
    """Build DBSCAN affinity clusters from co-purchase data.

    Pipeline:
    1. Query order-item pairs, group by order
    2. Build sparse NxN co-purchase matrix
    3. Reduce to 50D via TruncatedSVD
    4. Auto-tune eps and run DBSCAN to find affinity clusters
    5. Also fit a BallTree for backward-compatible neighbor lookups
    """
    print("  Building DBSCAN affinity clusters...")
    t0 = time.time()

    rows = session.execute(text("SELECT item_id, order_id FROM order_history")).fetchall()
    order_items: dict[str, list[int]] = defaultdict(list)
    for item_id, order_id in rows:
        order_items[order_id].append(item_id)

    all_item_ids = sorted(set(r[0] for r in rows))
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    n = len(all_item_ids)

    if n == 0:
        return AffinityIndex()

    print(f"  Building {n}x{n} co-purchase matrix from {len(order_items)} orders...")
    copurchase = lil_matrix((n, n), dtype=np.float32)
    for items_in_order in tqdm(order_items.values(), desc="  Co-purchase matrix"):
        mapped = [item_id_to_idx[i] for i in items_in_order if i in item_id_to_idx]
        for i in range(len(mapped)):
            for j in range(i + 1, len(mapped)):
                copurchase[mapped[i], mapped[j]] += 1
                copurchase[mapped[j], mapped[i]] += 1

    copurchase_csr = copurchase.tocsr()
    n_components = min(50, n - 1)
    print(f"  TruncatedSVD -> {n_components}D...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(copurchase_csr)
    reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)

    # Auto-tune eps and run DBSCAN
    eps = _auto_tune_dbscan_eps(reduced, k=min(5, n - 1))
    print(f"  DBSCAN with auto-tuned eps={eps:.3f}, min_samples=2...")
    db = DBSCAN(eps=eps, min_samples=2, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(reduced)

    # Build cluster membership dicts
    clusters: dict[int, list[int]] = defaultdict(list)
    noise_items = []
    for idx, label in enumerate(labels):
        item_id = int(all_item_ids[idx])
        if label == -1:
            noise_items.append(item_id)
        else:
            clusters[label].append(item_id)

    n_clusters = len(clusters)
    n_noise = len(noise_items)
    print(f"  DBSCAN found {n_clusters} clusters + {n_noise} noise items")
    if clusters:
        sizes = [len(v) for v in clusters.values()]
        print(f"    Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")

    # Also fit BallTree for backward-compat neighbor lookups
    nn_model = NearestNeighbors(
        n_neighbors=min(k_neighbors + 1, n),
        algorithm="ball_tree", metric="euclidean", leaf_size=40, n_jobs=-1,
    )
    nn_model.fit(reduced)

    elapsed = time.time() - t0
    print(f"  Affinity clusters built in {elapsed:.1f}s")

    return AffinityIndex(
        item_ids=np.array(all_item_ids, dtype=np.int64),
        item_id_to_idx=item_id_to_idx,
        cluster_labels=labels,
        clusters=dict(clusters),
        n_clusters=n_clusters,
        noise_items=noise_items,
        reduced_vectors=reduced,
        nn_model=nn_model,
        k_neighbors=k_neighbors,
    )


def get_affinity_neighbors(index: AffinityIndex, item_id: int) -> list[int]:
    """Backward-compatible: return K nearest neighbors via BallTree."""
    idx = index.item_id_to_idx.get(item_id)
    if idx is None or index.nn_model is None or index.reduced_vectors is None:
        return []
    vec = index.reduced_vectors[idx].reshape(1, -1)
    _, indices = index.nn_model.kneighbors(vec)
    return [int(index.item_ids[ni]) for ni in indices[0]
            if ni != idx and ni < len(index.item_ids)]


def get_cluster_members(index: AffinityIndex, item_id: int) -> list[int]:
    """Return all items in the same DBSCAN cluster as item_id."""
    idx = index.item_id_to_idx.get(item_id)
    if idx is None or index.cluster_labels is None:
        return []
    label = index.cluster_labels[idx]
    if label == -1:
        return []
    return index.clusters.get(int(label), [])


# ---------------------------------------------------------------------------
# Warehouse State — unchanged
# ---------------------------------------------------------------------------

@dataclass
class WarehouseState:
    """In-memory snapshot. slot_data: [id, x, y, w, h, d, cur_w, max_w, rack_id, occupied]"""
    slot_data: np.ndarray = None
    slot_zone_ids: np.ndarray = None
    slot_shelf_nums: np.ndarray = None
    slot_rack_shelves: np.ndarray = None
    slot_kdtree: KDTree = None
    zone_tags: dict[int, list[str]] = field(default_factory=dict)
    zone_distances: dict[int, float] = field(default_factory=dict)
    slot_idx_map: dict[int, int] = field(default_factory=dict)
    assignments: dict[int, int] = field(default_factory=dict)
    compatible_zones_cache: dict[str, np.ndarray] = field(default_factory=dict)
    warehouse_graph: WarehouseGraph = None
    slot_graph_distances: np.ndarray = None  # graph distance from each slot to picking area


def load_warehouse_state(session: Session) -> WarehouseState:
    """Load entire warehouse state into memory."""
    print("  Loading warehouse state into memory...")
    t0 = time.time()
    state = WarehouseState()

    for z in session.execute(text("SELECT id, tags_json, distance_to_picking_area FROM zones")).fetchall():
        state.zone_tags[z[0]] = json.loads(z[1])
        state.zone_distances[z[0]] = z[2]

    rack_zone, rack_shelves = {}, {}
    for r in session.execute(text("SELECT id, zone_id, num_shelves FROM racks")).fetchall():
        rack_zone[r[0]] = r[1]; rack_shelves[r[0]] = r[2]

    slots = session.execute(text("""
        SELECT s.id, s.x_coord, s.y_coord, s.width_cm, s.height_cm, s.depth_cm,
               s.current_weight_kg, s.max_weight_kg, s.rack_id,
               CASE WHEN i.id IS NOT NULL THEN 1 ELSE 0 END, s.shelf_number
        FROM slots s LEFT JOIN items i ON i.current_slot_id = s.id
    """)).fetchall()

    n = len(slots)
    slot_array = np.zeros((n, 10), dtype=np.float64)
    zone_ids = np.zeros(n, dtype=np.int64)
    shelf_nums = np.zeros(n, dtype=np.float64)
    rack_shelf_counts = np.zeros(n, dtype=np.float64)

    for i, s in enumerate(slots):
        slot_array[i] = s[:10]
        rid = int(s[8])
        zone_ids[i] = rack_zone.get(rid, 0)
        shelf_nums[i] = s[10]
        rack_shelf_counts[i] = rack_shelves.get(rid, 6)
        state.slot_idx_map[int(s[0])] = i

    state.slot_data = slot_array
    state.slot_zone_ids = zone_ids
    state.slot_shelf_nums = shelf_nums
    state.slot_rack_shelves = rack_shelf_counts
    if n > 0:
        state.slot_kdtree = KDTree(slot_array[:, 1:3])

    # Build the warehouse graph and compute graph-based distances
    state.warehouse_graph = build_warehouse_graph(session)
    graph_dists = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sid = int(slot_array[i, 0])
        graph_dists[i] = slot_to_picking_distance(state.warehouse_graph, sid)
    state.slot_graph_distances = graph_dists
    print(f"  Graph distances computed: min={graph_dists.min():.1f}m, max={graph_dists.max():.1f}m, avg={graph_dists.mean():.1f}m")

    elapsed = time.time() - t0
    print(f"  Loaded: {n} slots, {len(state.zone_tags)} zones ({elapsed:.1f}s)")
    return state


# ---------------------------------------------------------------------------
# Ergonomic Scoring — unchanged
# ---------------------------------------------------------------------------

def _ergonomic_score_vectorized(shelf_nums, rack_shelf_counts, item_weight):
    """Golden Zone scoring: middle shelves = 1.0, extremes = 0.1."""
    total = np.maximum(rack_shelf_counts, 2)
    norm_pos = (shelf_nums - 1) / (total - 1)
    deviation = np.abs(norm_pos - 0.4)
    ergo = np.clip(1.0 - deviation * 1.5, 0.1, 1.0)
    wf = min(1.0, item_weight / 5.0)
    return ergo * (0.5 + 0.5 * wf)


# ---------------------------------------------------------------------------
# Phase 2: Genetic Algorithm Optimizer
# ---------------------------------------------------------------------------

def _find_compatible_zones(item_tags: list[str], zone_tags: dict[int, list[str]]) -> list[int]:
    """Return zone IDs compatible with an item's tags."""
    return [zid for zid, ztags in zone_tags.items()
            if check_tag_compatibility(item_tags, ztags)]


def _ga_fitness(
    chromosome: np.ndarray,
    clusters: dict[int, list[int]],
    noise_zone_map: dict[int, int],
    item_data: dict[int, dict],
    zone_distances: dict[int, float],
    zone_slots_available: dict[int, int],
    zone_tags: dict[int, list[str]],
) -> float:
    """Evaluate a chromosome's fitness (lower = better).

    chromosome[i] = zone_id for cluster i.
    Fitness = total daily walking distance + constraint violation penalty.
    """
    total_walk = 0.0
    penalty = 0.0

    # Track how many items each zone would receive
    zone_item_count: dict[int, int] = defaultdict(int)

    # Evaluate cluster placements
    for cluster_id, zone_id in enumerate(chromosome):
        if cluster_id not in clusters:
            continue
        items = clusters[cluster_id]
        zone_dist = zone_distances.get(int(zone_id), 40.0)

        for item_id in items:
            idata = item_data.get(item_id)
            if not idata:
                continue
            daily_picks = idata["daily_picks"]
            total_walk += daily_picks * zone_dist * 2  # round trip

            # Safety tag violation = hard penalty (flammable/hazardous/chemical in wrong zone)
            if not check_tag_compatibility(idata["tags"], zone_tags.get(int(zone_id), [])):
                penalty += 100.0
            else:
                # Soft preference: bonus for matching tags, small penalty for mismatched
                pref = tag_preference_score(idata["tags"], zone_tags.get(int(zone_id), []))
                total_walk += (1.0 - pref) * 2.0  # slight walk penalty for non-matching zones

            zone_item_count[int(zone_id)] += 1

    # Evaluate noise items
    for item_id, zone_id in noise_zone_map.items():
        idata = item_data.get(item_id)
        if not idata:
            continue
        zone_dist = zone_distances.get(zone_id, 40.0)
        total_walk += idata["daily_picks"] * zone_dist * 2
        zone_item_count[zone_id] += 1

    # Capacity penalty: if a zone gets more items than it has slots
    for zid, count in zone_item_count.items():
        available = zone_slots_available.get(zid, 0)
        if count > available:
            penalty += (count - available) * 50.0

    return total_walk + penalty


def _ga_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Uniform crossover: each gene has 50% chance from either parent."""
    mask = np.random.random(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child


def _ga_mutate(
    chromosome: np.ndarray,
    cluster_compatible_zones: dict[int, list[int]],
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """Mutate: randomly reassign clusters to compatible zones."""
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if py_random.random() < mutation_rate:
            compat = cluster_compatible_zones.get(i, [])
            if compat:
                mutated[i] = py_random.choice(compat)
    return mutated


def optimize_layout_ga(
    affinity_index: AffinityIndex,
    item_data: dict[int, dict],
    zone_distances: dict[int, float],
    zone_slots_available: dict[int, int],
    zone_tags: dict[int, list[str]],
    pop_size: int = 50,
    n_generations: int = 100,
    elitism: int = 2,
) -> tuple[np.ndarray, dict[int, int]]:
    """Run genetic algorithm to find optimal cluster-to-zone assignment.

    Returns (best_chromosome, noise_zone_map).
    """
    print(f"  Running Genetic Algorithm ({pop_size} pop x {n_generations} gen)...")
    t0 = time.time()

    clusters = affinity_index.clusters
    n_clusters = len(clusters)
    zone_ids = list(zone_distances.keys())

    if n_clusters == 0 or not zone_ids:
        return np.array([], dtype=np.int64), {}

    # Pre-compute compatible zones for each cluster
    cluster_compatible: dict[int, list[int]] = {}
    for cid, items in clusters.items():
        all_tags = set()
        for iid in items:
            idata = item_data.get(iid)
            if idata:
                all_tags.update(idata["tags"])
        compat = [zid for zid in zone_ids
                   if check_tag_compatibility(list(all_tags), zone_tags.get(zid, []))]
        if not compat:
            compat = zone_ids[:]
        cluster_compatible[cid] = compat

    # Assign noise items to their best compatible zone (by velocity)
    noise_zone_map: dict[int, int] = {}
    for item_id in affinity_index.noise_items:
        idata = item_data.get(item_id)
        if not idata:
            continue
        compat = _find_compatible_zones(idata["tags"], zone_tags) or zone_ids
        if idata["velocity_class"] == "A":
            best = min(compat, key=lambda z: zone_distances.get(z, 99))
        elif idata["velocity_class"] == "C":
            best = max(compat, key=lambda z: zone_distances.get(z, 0))
        else:
            best = compat[len(compat) // 2]
        noise_zone_map[item_id] = best

    # Initialize population
    population = []
    for _ in range(pop_size):
        chrom = np.zeros(n_clusters, dtype=np.int64)
        for cid in range(n_clusters):
            compat = cluster_compatible.get(cid, zone_ids)
            chrom[cid] = py_random.choice(compat)
        population.append(chrom)

    # Also seed one "smart" individual: assign clusters to nearest compatible zone
    # weighted by average velocity of cluster members
    smart = np.zeros(n_clusters, dtype=np.int64)
    for cid, items in clusters.items():
        avg_vel = np.mean([item_data.get(iid, {}).get("velocity_score", 0) for iid in items])
        compat = cluster_compatible.get(cid, zone_ids)
        if avg_vel > 0.5:
            smart[cid] = min(compat, key=lambda z: zone_distances.get(z, 99))
        elif avg_vel < 0.2:
            smart[cid] = max(compat, key=lambda z: zone_distances.get(z, 0))
        else:
            smart[cid] = compat[len(compat) // 2]
    population[0] = smart

    # Evolve
    best_fitness = float("inf")
    best_chrom = population[0].copy()
    stagnant = 0

    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = np.array([
            _ga_fitness(ch, clusters, noise_zone_map, item_data,
                        zone_distances, zone_slots_available, zone_tags)
            for ch in population
        ])

        # Track best
        gen_best_idx = np.argmin(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_chrom = population[gen_best_idx].copy()
            stagnant = 0
        else:
            stagnant += 1

        if gen % 20 == 0 or gen == n_generations - 1:
            print(f"    Gen {gen:3d}: best fitness = {best_fitness:.1f} (walk distance)")

        # Early stop if stagnant
        if stagnant > 30:
            print(f"    Early stop at gen {gen} (stagnant for 30 gens)")
            break

        # Selection + breeding
        sorted_indices = np.argsort(fitnesses)
        new_pop = [population[sorted_indices[i]].copy() for i in range(elitism)]

        while len(new_pop) < pop_size:
            # Tournament selection
            t_size = min(3, len(population))
            t_indices = py_random.sample(range(len(population)), t_size)
            parent1 = population[min(t_indices, key=lambda i: fitnesses[i])]
            t_indices = py_random.sample(range(len(population)), t_size)
            parent2 = population[min(t_indices, key=lambda i: fitnesses[i])]

            child = _ga_crossover(parent1, parent2)
            child = _ga_mutate(child, cluster_compatible, mutation_rate=0.1)
            new_pop.append(child)

        population = new_pop[:pop_size]

    elapsed = time.time() - t0
    print(f"  GA complete in {elapsed:.1f}s. Best fitness: {best_fitness:.1f}")

    return best_chrom, noise_zone_map


# ---------------------------------------------------------------------------
# Phase 3: Assign items to actual slots from GA solution
# ---------------------------------------------------------------------------

def assign_items_to_slots(
    session: Session,
    state: WarehouseState,
    affinity_index: AffinityIndex,
    velocity_map: dict[int, tuple[float, str]],
    batch_size: int = 2000,
    score_weights: dict[str, float] | None = None,
    candidate_limit: int = 500,
) -> dict:
    """Assign items using DBSCAN clusters + GA-optimized zone mapping.

    1. Build item_data dict with tags, velocity, dimensions, daily picks
    2. Compute zone slot availability
    3. Run GA to find optimal cluster-to-zone mapping
    4. For each cluster's assigned zone, place items in slots (velocity-sorted)
    5. Persist assignments
    """
    print("\n  === DBSCAN + GA Slot Assignment ===")

    items = session.execute(text("""
        SELECT id, tags_json, weight_kg, width_cm, height_cm, depth_cm,
               velocity_score, velocity_class
        FROM items WHERE current_slot_id IS NULL
        ORDER BY velocity_score DESC
    """)).fetchall()

    if not items:
        return {"total_assigned": 0, "total_failed": 0, "total_items": 0,
                "A": 0, "B": 0, "C": 0, "failed_tags": 0, "failed_capacity": 0,
                "relaxation_counts": {}, "exceptions": []}

    # Build item_data dict
    item_data: dict[int, dict] = {}
    for row in items:
        iid = int(row[0])
        oc = velocity_map.get(iid, (0, "C"))
        item_data[iid] = {
            "tags": json.loads(row[1]), "weight": float(row[2]),
            "w": float(row[3]), "h": float(row[4]), "d": float(row[5]),
            "velocity_score": float(row[6]), "velocity_class": row[7],
            "daily_picks": oc[0] * 2.0,  # normalized score * 2 as proxy for daily picks
        }

    # Compute zone slot availability and graph-based zone distances
    zone_slots_available: dict[int, int] = {}
    graph_zone_distances: dict[int, float] = {}
    for zid in state.zone_tags:
        mask = (state.slot_zone_ids == zid) & (state.slot_data[:, 9] == 0)
        zone_slots_available[zid] = int(np.sum(mask))
        # Use average graph distance for slots in this zone (more accurate than flat zone distance)
        zone_mask = state.slot_zone_ids == zid
        if state.slot_graph_distances is not None and np.sum(zone_mask) > 0:
            graph_zone_distances[zid] = float(np.mean(state.slot_graph_distances[zone_mask]))
        else:
            graph_zone_distances[zid] = state.zone_distances.get(zid, 40.0)

    print(f"  Items to assign: {len(items)}")
    print(f"  DBSCAN clusters: {affinity_index.n_clusters} + {len(affinity_index.noise_items)} noise")
    print(f"  Zones: {list(state.zone_distances.keys())}")
    print(f"  Available slots per zone: {zone_slots_available}")
    print(f"  Graph-based zone distances: {graph_zone_distances}")

    # Run GA with graph-based distances (more accurate than flat zone distances)
    best_chrom, noise_zone_map = optimize_layout_ga(
        affinity_index, item_data,
        graph_zone_distances, zone_slots_available,
        state.zone_tags,
        pop_size=50, n_generations=100,
    )

    # Build item -> target_zone mapping from GA solution
    item_target_zone: dict[int, int] = {}

    # Cluster items
    for cluster_id, zone_id in enumerate(best_chrom):
        if cluster_id in affinity_index.clusters:
            for item_id in affinity_index.clusters[cluster_id]:
                item_target_zone[item_id] = int(zone_id)

    # Noise items
    for item_id, zone_id in noise_zone_map.items():
        item_target_zone[item_id] = zone_id

    # Items not in any cluster or noise (those without order history)
    for row in items:
        iid = int(row[0])
        if iid not in item_target_zone:
            tags = json.loads(row[1])
            compat = _find_compatible_zones(tags, state.zone_tags)
            if compat:
                item_target_zone[iid] = compat[0]
            else:
                item_target_zone[iid] = list(state.zone_tags.keys())[0] if state.zone_tags else 0

    # --- Assign items to actual slots ---
    print(f"  Assigning {len(items)} items to slots based on GA solution...")

    max_zone_dist = max(state.zone_distances.values()) if state.zone_distances else 1.0
    zone_dist_arr = np.array([
        state.zone_distances.get(int(zid), max_zone_dist)
        for zid in state.slot_zone_ids
    ], dtype=np.float64)

    assigned = 0
    failed = 0
    stats = {"A": 0, "B": 0, "C": 0, "failed_tags": 0, "failed_capacity": 0,
             "relaxation_counts": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, "exceptions": []}
    update_batch = []

    # Sort items: A first (they get priority for near-picking slots)
    sorted_items = sorted(items, key=lambda r: -float(r[6]))

    for item_row in tqdm(sorted_items, desc="  Placing items"):
        item_id = int(item_row[0])
        item_tags = json.loads(item_row[1])
        item_weight = float(item_row[2])
        item_w, item_h, item_d = float(item_row[3]), float(item_row[4]), float(item_row[5])
        item_velocity = float(item_row[6])
        item_class = item_row[7]

        # Filter to compatible slots (strict first)
        candidate_idx = filter_compatible_slots_vectorized(
            item_tags, item_weight, item_w, item_h, item_d,
            state.slot_data, state.slot_zone_ids,
            state.zone_tags, state.compatible_zones_cache,
        )
        relax_level = 0

        if len(candidate_idx) == 0:
            candidate_idx, relax_level = filter_with_relaxation(
                item_tags, item_weight, item_w, item_h, item_d,
                state.slot_data, state.slot_zone_ids,
                state.zone_tags, state.compatible_zones_cache,
            )

        if len(candidate_idx) == 0:
            failed += 1
            unoccupied = np.sum(state.slot_data[:, 9] == 0)
            stats["failed_capacity" if unoccupied == 0 else "failed_tags"] += 1
            continue

        stats["relaxation_counts"][relax_level] = stats["relaxation_counts"].get(relax_level, 0) + 1
        if relax_level > 0:
            stats["exceptions"].append({"item": item_id, "level": relax_level,
                                         "type": {1: "dimension", 2: "zone", 3: "weight", 4: "oversize", 5: "forced"}.get(relax_level, "?")})

        # Prefer slots in the GA-assigned target zone
        target_zone = item_target_zone.get(item_id)
        if target_zone is not None:
            in_target = np.array([i for i in candidate_idx
                                  if state.slot_zone_ids[i] == target_zone], dtype=np.int64)
            if len(in_target) > 0:
                candidate_idx = in_target

        # Score candidates: proximity (graph-based) + ergonomic
        n_cand = len(candidate_idx)
        scores = np.zeros(n_cand, dtype=np.float64)

        # Proximity using GRAPH distances (Dijkstra, not Euclidean)
        if state.slot_graph_distances is not None:
            graph_dists = state.slot_graph_distances[candidate_idx]
            max_gd = graph_dists.max() if len(graph_dists) > 0 else 1.0
            if max_gd > 0:
                scores += 0.6 * (1.0 - graph_dists / max_gd) * item_velocity
        elif max_zone_dist > 0:
            dists = zone_dist_arr[candidate_idx]
            scores += 0.6 * (1.0 - dists / max_zone_dist) * item_velocity

        # Ergonomic golden zone
        ergo = _ergonomic_score_vectorized(
            state.slot_shelf_nums[candidate_idx],
            state.slot_rack_shelves[candidate_idx],
            item_weight,
        )
        scores += 0.4 * ergo

        best_local = np.argmax(scores)
        best_global_idx = candidate_idx[best_local]
        best_slot_id = int(state.slot_data[best_global_idx, 0])

        # Assign
        state.slot_data[best_global_idx, 6] += item_weight
        state.slot_data[best_global_idx, 9] = 1
        state.assignments[item_id] = best_slot_id
        assigned += 1
        stats[item_class] += 1

        update_batch.append((item_id, best_slot_id, item_weight))
        if len(update_batch) >= batch_size:
            _persist_assignments(session, update_batch)
            update_batch = []

    if update_batch:
        _persist_assignments(session, update_batch)

    session.commit()
    stats.update({"total_assigned": assigned, "total_failed": failed, "total_items": len(items)})

    print(f"  Assignment complete: {assigned} placed, {failed} failed")
    print(f"    A: {stats['A']} | B: {stats['B']} | C: {stats['C']}")

    return stats


def _persist_assignments(session: Session, batch: list[tuple[int, int, float]]) -> None:
    """Persist a batch of item-to-slot assignments."""
    for item_id, slot_id, weight in batch:
        session.execute(text("UPDATE items SET current_slot_id=:sid WHERE id=:iid"),
                        {"sid": slot_id, "iid": item_id})
        session.execute(text("UPDATE slots SET current_weight_kg=current_weight_kg+:w WHERE id=:sid"),
                        {"w": weight, "sid": slot_id})
    session.flush()


# ---------------------------------------------------------------------------
# Analytics — unchanged
# ---------------------------------------------------------------------------

def compute_assignment_metrics(session: Session, state: WarehouseState) -> dict:
    """Compute quality metrics for the current slot assignments."""
    result = session.execute(text("""
        SELECT i.velocity_class, COUNT(*) as cnt,
               AVG(z.distance_to_picking_area) as avg_dist
        FROM items i JOIN slots s ON i.current_slot_id=s.id
        JOIN racks r ON s.rack_id=r.id JOIN zones z ON r.zone_id=z.id
        WHERE i.current_slot_id IS NOT NULL
        GROUP BY i.velocity_class ORDER BY i.velocity_class
    """)).fetchall()

    metrics = {}
    for row in result:
        metrics[f"class_{row[0]}_count"] = row[1]
        metrics[f"class_{row[0]}_avg_distance"] = round(row[2], 2)

    total = session.execute(text("SELECT COUNT(*) FROM items")).scalar()
    assigned = session.execute(text("SELECT COUNT(*) FROM items WHERE current_slot_id IS NOT NULL")).scalar()
    metrics["assignment_rate"] = round(assigned / total * 100, 1) if total > 0 else 0
    metrics["total_items"] = total
    metrics["assigned_items"] = assigned

    total_slots = session.execute(text("SELECT COUNT(*) FROM slots")).scalar()
    used = session.execute(text("SELECT COUNT(DISTINCT current_slot_id) FROM items WHERE current_slot_id IS NOT NULL")).scalar()
    metrics["slot_utilization"] = round(used / total_slots * 100, 1) if total_slots > 0 else 0
    metrics["total_slots"] = total_slots
    metrics["used_slots"] = used

    return metrics

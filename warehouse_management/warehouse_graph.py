"""Warehouse graph model — represents the floor as a walkable graph.

Nodes = aisle intersections, rack endpoints, picking area, zone entries
Edges = walkable paths between nodes, weighted by real walking distance

This replaces naive Euclidean distance with Dijkstra shortest-path
distance that respects aisles, walls, and rack barriers.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class WarehouseGraph:
    """Graph representation of the warehouse floor layout."""
    graph: nx.Graph = field(default_factory=nx.Graph)
    slot_to_node: dict[int, str] = field(default_factory=dict)
    rack_to_node: dict[int, str] = field(default_factory=dict)
    zone_to_node: dict[int, str] = field(default_factory=dict)
    picking_node: str = "PICKING"
    _dist_cache: dict[tuple[str, str], float] = field(default_factory=dict)


def build_warehouse_graph(session: Session) -> WarehouseGraph:
    """Build a NetworkX graph from the warehouse database.

    Layout structure:
    - PICKING node at (0, 0)
    - Each zone has an entry node connected to PICKING (edge weight = zone distance)
    - Each rack has a node at its center, connected to its zone entry
    - Each slot connects to its rack node
    - Adjacent racks within a zone are connected (aisle paths)
    - Cross-zone connections through a main aisle corridor

    The graph captures real walking paths: to go from Rack A to Rack B,
    you walk along the aisle to a junction, then down another aisle —
    not straight through the rack shelves.
    """
    print("  Building warehouse graph...")
    wg = WarehouseGraph()
    G = wg.graph

    # Add picking area node
    G.add_node("PICKING", x=0, y=0, type="picking")

    # Load zones
    zones = session.execute(text(
        "SELECT id, name, distance_to_picking_area, tags_json FROM zones ORDER BY distance_to_picking_area"
    )).fetchall()

    zone_data = {}
    for z in zones:
        zid, zname, zdist, ztags = z
        zone_node = f"ZONE_{zid}"
        G.add_node(zone_node, x=50, y=zdist * 3, type="zone", zone_id=zid, name=zname)
        # Connect zone entry to picking area (weight = actual walking distance)
        G.add_edge("PICKING", zone_node, weight=zdist)
        wg.zone_to_node[zid] = zone_node
        zone_data[zid] = {"name": zname, "dist": zdist, "y_offset": zdist * 3}

    # Connect zones to each other via a main corridor
    zone_nodes = [wg.zone_to_node[z[0]] for z in zones]
    for i in range(len(zone_nodes) - 1):
        dist = abs(zone_data[zones[i][0]]["dist"] - zone_data[zones[i+1][0]]["dist"])
        G.add_edge(zone_nodes[i], zone_nodes[i+1], weight=dist)

    # Load racks and create rack nodes + aisle connections
    racks = session.execute(text(
        "SELECT id, zone_id, name FROM racks ORDER BY zone_id, id"
    )).fetchall()

    racks_by_zone: dict[int, list] = defaultdict(list)
    for r in racks:
        rid, zid, rname = r
        racks_by_zone[zid].append((rid, rname))

    for zid, zone_racks in racks_by_zone.items():
        zone_node = wg.zone_to_node.get(zid)
        if not zone_node:
            continue

        base_y = zone_data[zid]["y_offset"]

        for idx, (rid, rname) in enumerate(zone_racks):
            rack_node = f"RACK_{rid}"
            # Racks laid out side by side within the zone
            rx = 20 + idx * 10
            ry = base_y
            G.add_node(rack_node, x=rx, y=ry, type="rack", rack_id=rid, name=rname)
            wg.rack_to_node[rid] = rack_node

            # Connect rack to zone entry (walking distance = aisle distance)
            aisle_dist = 3.0 + idx * 2.0  # further racks are deeper into the zone
            G.add_edge(zone_node, rack_node, weight=aisle_dist)

            # Connect to adjacent racks (walking between racks in the same aisle)
            if idx > 0:
                prev_rid = zone_racks[idx - 1][0]
                prev_node = f"RACK_{prev_rid}"
                G.add_edge(prev_node, rack_node, weight=2.0)  # adjacent racks are 2m apart

    # Load slots and connect to rack nodes
    slots = session.execute(text(
        "SELECT id, rack_id, shelf_number, position_on_shelf, x_coord, y_coord FROM slots"
    )).fetchall()

    for s in slots:
        sid, rid, shelf, pos, sx, sy = s
        slot_node = f"SLOT_{sid}"
        G.add_node(slot_node, x=sx, y=sy, type="slot", slot_id=sid, shelf=shelf, position=pos)
        wg.slot_to_node[sid] = slot_node

        rack_node = wg.rack_to_node.get(rid)
        if rack_node:
            # Walking distance within the rack: reach + shelf height factor
            reach_dist = 0.5 + abs(shelf - 3) * 0.3  # middle shelves are easier to reach
            G.add_edge(rack_node, slot_node, weight=reach_dist)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  Graph built: {n_nodes} nodes, {n_edges} edges")
    print(f"    Zones: {len(zone_data)}, Racks: {len(racks)}, Slots: {len(slots)}")

    return wg


def graph_distance(wg: WarehouseGraph, node_a: str, node_b: str) -> float:
    """Compute shortest-path distance between two nodes using Dijkstra.

    Uses a cache to avoid recomputing frequently-used paths.
    """
    if node_a == node_b:
        return 0.0

    cache_key = (min(node_a, node_b), max(node_a, node_b))
    if cache_key in wg._dist_cache:
        return wg._dist_cache[cache_key]

    try:
        dist = nx.shortest_path_length(wg.graph, node_a, node_b, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist = 999.0

    wg._dist_cache[cache_key] = dist
    return dist


def slot_to_picking_distance(wg: WarehouseGraph, slot_id: int) -> float:
    """Get the shortest walking distance from a slot to the picking area."""
    slot_node = wg.slot_to_node.get(slot_id)
    if not slot_node:
        return 999.0
    return graph_distance(wg, slot_node, wg.picking_node)


def slot_to_slot_distance(wg: WarehouseGraph, slot_a: int, slot_b: int) -> float:
    """Get the shortest walking distance between two slots."""
    na = wg.slot_to_node.get(slot_a)
    nb = wg.slot_to_node.get(slot_b)
    if not na or not nb:
        return 999.0
    return graph_distance(wg, na, nb)


def get_graph_stats(wg: WarehouseGraph) -> dict:
    """Return summary statistics about the warehouse graph."""
    G = wg.graph
    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("type", "unknown")] += 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": dict(node_types),
        "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
        "avg_degree": sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1),
    }


def get_shortest_path(wg: WarehouseGraph, node_a: str, node_b: str) -> tuple[list[str], float]:
    """Return the shortest path and its total distance between two nodes."""
    try:
        path = nx.shortest_path(wg.graph, node_a, node_b, weight="weight")
        dist = nx.shortest_path_length(wg.graph, node_a, node_b, weight="weight")
        return path, dist
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], 999.0


def get_graph_for_visualization(wg: WarehouseGraph) -> dict:
    """Return graph data formatted for dashboard visualization."""
    G = wg.graph
    nodes = []
    for nid, data in G.nodes(data=True):
        nodes.append({
            "id": nid,
            "x": data.get("x", 0),
            "y": data.get("y", 0),
            "type": data.get("type", "unknown"),
            "name": data.get("name", nid),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "from": u, "to": v,
            "weight": round(data.get("weight", 1.0), 1),
        })

    return {"nodes": nodes, "edges": edges}

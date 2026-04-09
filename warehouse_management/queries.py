"""Reusable DB query functions returning DataFrames for the dashboard."""

from __future__ import annotations

import json
import math
import random

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from .test_data import AFFINITY_GROUPS


def _build_item_group_map() -> dict[str, str]:
    """Return {item_name: group_name} for all affinity group items."""
    m = {}
    for gname, group in AFFINITY_GROUPS.items():
        label = gname.replace("_", " ").title()
        for iname in group["items"]:
            m[iname] = label
    return m


# Each affinity group gets a distinctive border color for visual clustering
AFFINITY_GROUP_COLORS = {
    "Breakfast": "#f59e0b",
    "Pasta Dinner": "#f97316",
    "Baking": "#a855f7",
    "Snacks": "#ec4899",
    "Beverages": "#3b82f6",
    "Dairy Cold": "#06b6d4",
    "Cleaning": "#8b5cf6",
    "Personal Care": "#14b8a6",
    "Flammable Goods": "#ef4444",
    "Electronics Acc": "#6366f1",
    "Clothing Basics": "#ec4899",
    "Heavy Tools": "#f97316",
}


def get_warehouse_compact_layout(session: Session) -> tuple[list[dict], list[str]]:
    """Return zones with racks laid out in a grid, each rack being a column of shelf rows.

    Returns (layout, group_names).
    Each zone has a 'racks' list; each rack has a 'shelves' dict mapping
    shelf_number -> list of slot dicts (one per position on that shelf).
    This lets the UI render a 2D floor-plan grid.
    """
    item_group_map = _build_item_group_map()

    zones_raw = session.execute(text("""
        SELECT id, name, tags_json, distance_to_picking_area
        FROM zones ORDER BY distance_to_picking_area
    """)).fetchall()

    groups_found = set()
    layout = []
    for z in zones_raw:
        zone_id, zone_name, tags_json, distance = z
        tags = json.loads(tags_json)

        racks_raw = session.execute(text("""
            SELECT id, name, num_shelves FROM racks
            WHERE zone_id = :zid ORDER BY id
        """), {"zid": zone_id}).fetchall()

        zone_total = zone_occ = zone_a = zone_b = zone_c = 0
        racks_list = []

        for rack_row in racks_raw:
            rid, rname, num_sh = rack_row
            slots_raw = session.execute(text("""
                SELECT s.id, s.shelf_number, s.position_on_shelf,
                       i.name, i.sku, i.velocity_class, i.weight_kg
                FROM slots s
                LEFT JOIN items i ON i.current_slot_id = s.id
                WHERE s.rack_id = :rid
                ORDER BY s.shelf_number, s.position_on_shelf
            """), {"rid": rid}).fetchall()

            shelves = {}
            for sl in slots_raw:
                occupied = sl[3] is not None
                vc = sl[5] if occupied else ""
                grp = ""
                zone_total += 1
                if occupied:
                    zone_occ += 1
                    if vc == "A": zone_a += 1
                    elif vc == "B": zone_b += 1
                    else: zone_c += 1
                    grp = item_group_map.get(sl[3], "")
                    if grp:
                        groups_found.add(grp)
                slot_d = {
                    "id": sl[0], "occ": occupied, "vc": vc,
                    "name": sl[3] or "", "sku": sl[4] or "",
                    "w": sl[6] or 0, "grp": grp,
                }
                shelves.setdefault(sl[1], []).append(slot_d)

            racks_list.append({
                "id": rid, "name": rname, "num_shelves": num_sh,
                "shelves": shelves,
                "slot_count": len(slots_raw),
            })

        layout.append({
            "id": zone_id, "name": zone_name, "tags": tags,
            "distance": distance, "rack_count": len(racks_list),
            "racks": racks_list,
            "total": zone_total, "occ": zone_occ,
            "a": zone_a, "b": zone_b, "c": zone_c,
        })
    return layout, sorted(groups_found)


def get_rack_detail(session: Session, rack_id: int) -> pd.DataFrame:
    result = session.execute(text("""
        SELECT s.id AS slot_id, s.shelf_number, s.position_on_shelf,
               s.width_cm, s.height_cm, s.depth_cm,
               s.current_weight_kg, s.max_weight_kg,
               i.id AS item_id, i.sku, i.name AS item_name, i.velocity_class,
               i.weight_kg AS item_weight,
               i.width_cm AS item_w, i.height_cm AS item_h, i.depth_cm AS item_d
        FROM slots s LEFT JOIN items i ON i.current_slot_id = s.id
        WHERE s.rack_id = :rack_id
        ORDER BY s.shelf_number, s.position_on_shelf
    """), {"rack_id": rack_id})
    rows = result.fetchall()
    columns = [
        "slot_id", "shelf_number", "position_on_shelf",
        "width_cm", "height_cm", "depth_cm",
        "current_weight_kg", "max_weight_kg",
        "item_id", "sku", "item_name", "velocity_class",
        "item_weight", "item_w", "item_h", "item_d",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["occupied"] = df["item_id"].notna()
    df["weight_pct"] = (df["current_weight_kg"] / df["max_weight_kg"] * 100).clip(0, 100)
    return df


def get_zone_summary(session: Session) -> pd.DataFrame:
    result = session.execute(text("""
        SELECT z.id AS zone_id, z.name AS zone_name, z.tags_json,
               z.distance_to_picking_area,
               COUNT(DISTINCT r.id) AS rack_count,
               COUNT(DISTINCT s.id) AS slot_count,
               COUNT(DISTINCT i.id) AS item_count,
               ROUND(CAST(COUNT(DISTINCT i.id) AS REAL)/NULLIF(COUNT(DISTINCT s.id),0)*100,1) AS utilization_pct
        FROM zones z LEFT JOIN racks r ON r.zone_id=z.id
        LEFT JOIN slots s ON s.rack_id=r.id
        LEFT JOIN items i ON i.current_slot_id=s.id
        GROUP BY z.id ORDER BY z.distance_to_picking_area
    """))
    cols = ["zone_id","zone_name","tags_json","distance_to_picking_area",
            "rack_count","slot_count","item_count","utilization_pct"]
    return pd.DataFrame(result.fetchall(), columns=cols)


def get_kpi_metrics(session: Session) -> dict:
    ti = session.execute(text("SELECT COUNT(*) FROM items")).scalar()
    ai = session.execute(text("SELECT COUNT(*) FROM items WHERE current_slot_id IS NOT NULL")).scalar()
    ts = session.execute(text("SELECT COUNT(*) FROM slots")).scalar()
    us = session.execute(text("SELECT COUNT(DISTINCT current_slot_id) FROM items WHERE current_slot_id IS NOT NULL")).scalar()
    to = session.execute(text("SELECT COUNT(DISTINCT order_id) FROM order_history")).scalar()
    tz = session.execute(text("SELECT COUNT(*) FROM zones")).scalar()
    tr = session.execute(text("SELECT COUNT(*) FROM racks")).scalar()
    return {
        "total_items": ti, "assigned_items": ai, "unassigned_items": ti-ai,
        "assignment_rate": round(ai/ti*100,1) if ti else 0,
        "total_slots": ts, "used_slots": us,
        "slot_utilization": round(us/ts*100,1) if ts else 0,
        "total_orders": to, "total_zones": tz, "total_racks": tr,
    }


def get_velocity_distribution(session: Session) -> pd.DataFrame:
    result = session.execute(text("""
        SELECT velocity_class, COUNT(*) as count,
               SUM(CASE WHEN current_slot_id IS NOT NULL THEN 1 ELSE 0 END) as assigned
        FROM items GROUP BY velocity_class ORDER BY velocity_class
    """))
    return pd.DataFrame(result.fetchall(), columns=["velocity_class","count","assigned"])


def get_distance_by_class(session: Session) -> pd.DataFrame:
    result = session.execute(text("""
        SELECT i.velocity_class, ROUND(AVG(z.distance_to_picking_area),1) as avg_distance,
               COUNT(*) as item_count
        FROM items i JOIN slots s ON i.current_slot_id=s.id
        JOIN racks r ON s.rack_id=r.id JOIN zones z ON r.zone_id=z.id
        WHERE i.current_slot_id IS NOT NULL
        GROUP BY i.velocity_class ORDER BY i.velocity_class
    """))
    return pd.DataFrame(result.fetchall(), columns=["velocity_class","avg_distance","item_count"])


def get_top_items(session: Session, n: int = 10) -> pd.DataFrame:
    result = session.execute(text("""
        SELECT i.sku, i.name, i.velocity_class, COUNT(DISTINCT oh.order_id) AS oc,
               z.name, r.name, s.shelf_number
        FROM items i JOIN order_history oh ON oh.item_id=i.id
        LEFT JOIN slots s ON i.current_slot_id=s.id
        LEFT JOIN racks r ON s.rack_id=r.id LEFT JOIN zones z ON r.zone_id=z.id
        GROUP BY i.id ORDER BY oc DESC LIMIT :n
    """), {"n": n})
    return pd.DataFrame(result.fetchall(), columns=[
        "sku","item_name","velocity_class","order_count","zone_name","rack_name","shelf_number"])


def get_zone_item_distribution(session: Session) -> pd.DataFrame:
    result = session.execute(text("""
        SELECT z.name, COUNT(i.id) FROM zones z
        LEFT JOIN racks r ON r.zone_id=z.id LEFT JOIN slots s ON s.rack_id=r.id
        LEFT JOIN items i ON i.current_slot_id=s.id
        GROUP BY z.id HAVING COUNT(i.id)>0 ORDER BY COUNT(i.id) DESC
    """))
    return pd.DataFrame(result.fetchall(), columns=["zone_name","item_count"])


def get_zones_list(session: Session) -> pd.DataFrame:
    result = session.execute(text("SELECT id, name FROM zones ORDER BY distance_to_picking_area"))
    return pd.DataFrame(result.fetchall(), columns=["id","name"])


def get_racks_for_zone(session: Session, zone_id: int) -> pd.DataFrame:
    result = session.execute(text(
        "SELECT id, name FROM racks WHERE zone_id=:zid ORDER BY name"
    ), {"zid": zone_id})
    return pd.DataFrame(result.fetchall(), columns=["id","name"])


def get_rack_info(session: Session, rack_id: int) -> dict:
    row = session.execute(text("""
        SELECT r.name, r.max_weight_kg, r.num_shelves,
               r.shelf_height_cm, r.shelf_width_cm, r.shelf_depth_cm, z.name
        FROM racks r JOIN zones z ON r.zone_id=z.id WHERE r.id=:rid
    """), {"rid": rack_id}).fetchone()
    if not row:
        return {}
    return {"name": row[0], "max_weight_kg": row[1], "num_shelves": row[2],
            "shelf_height_cm": row[3], "shelf_width_cm": row[4],
            "shelf_depth_cm": row[5], "zone_name": row[6]}


def get_affinity_group_placement(session: Session) -> list[dict]:
    results = []
    for gname, group in AFFINITY_GROUPS.items():
        names = group["items"]
        ph = ",".join([f"'{n}'" for n in names])
        rows = session.execute(text(f"""
            SELECT i.name, i.sku, i.velocity_class, s.x_coord, s.y_coord,
                   z.name, r.name, i.current_slot_id
            FROM items i LEFT JOIN slots s ON i.current_slot_id=s.id
            LEFT JOIN racks r ON s.rack_id=r.id LEFT JOIN zones z ON r.zone_id=z.id
            WHERE i.name IN ({ph})
        """)).fetchall()

        items_info, coords, placed_count, zones_used = [], [], 0, set()
        for r in rows:
            placed = r[7] is not None
            if placed:
                placed_count += 1
                if r[3] is not None:
                    coords.append((r[3], r[4]))
                if r[5]:
                    zones_used.add(r[5])
            items_info.append({"name": r[0], "sku": r[1], "vc": r[2],
                               "zone": r[5] or "—", "rack": r[6] or "—", "placed": placed})

        avg_dist = 0.0
        if len(coords) >= 2:
            ds = [math.sqrt((coords[i][0]-coords[j][0])**2+(coords[i][1]-coords[j][1])**2)
                  for i in range(len(coords)) for j in range(i+1, len(coords))]
            avg_dist = sum(ds)/len(ds) if ds else 0

        cs = max(0, min(100, (1-avg_dist/100)*100)) if placed_count > 0 and len(coords) >= 2 else 0.0
        results.append({
            "group": gname.replace("_"," ").title(), "tags": group["tags"],
            "items": items_info, "total": len(names), "placed": placed_count,
            "zones_used": list(zones_used), "avg_distance": round(avg_dist,1),
            "cluster_score": round(cs,1),
        })
    return results


def add_items_to_db(session: Session, items_data: list[dict]) -> int:
    mx = session.execute(text("SELECT MAX(CAST(SUBSTR(sku,5) AS INTEGER)) FROM items")).scalar() or 0
    count = 0
    for i, item in enumerate(items_data):
        sn = mx + i + 1
        session.execute(text("""
            INSERT INTO items (sku,name,tags_json,weight_kg,width_cm,height_cm,depth_cm,velocity_score,velocity_class)
            VALUES (:sku,:name,:tags,:w,:wi,:hi,:di,0.0,'C')
        """), {"sku": f"SKU-{sn:06d}", "name": item["name"], "tags": json.dumps(item["tags"]),
               "w": item["weight_kg"], "wi": item["width_cm"], "hi": item["height_cm"], "di": item["depth_cm"]})
        count += 1
    session.commit()
    return count


def add_racks_to_db(session: Session, zone_id: int, count: int,
                    num_shelves: int, shelf_width: float, shelf_height: float, shelf_depth: float) -> int:
    zone = session.execute(text("SELECT name FROM zones WHERE id=:zid"), {"zid": zone_id}).fetchone()
    if not zone:
        return 0
    erc = session.execute(text("SELECT COUNT(*) FROM racks WHERE zone_id=:zid"), {"zid": zone_id}).scalar()
    max_weight = 200.0
    pps = max(1, int(shelf_width/25))
    wps = round(max_weight/num_shelves/pps, 2)
    for i in range(count):
        rn = erc + i + 1
        rname = f"{zone[0]}-R{rn:03d}"
        session.execute(text("""
            INSERT INTO racks (zone_id,name,max_weight_kg,num_shelves,shelf_height_cm,shelf_width_cm,shelf_depth_cm)
            VALUES (:zid,:name,:mw,:ns,:sh,:sw,:sd)
        """), {"zid": zone_id, "name": rname, "mw": max_weight,
               "ns": num_shelves, "sh": shelf_height, "sw": shelf_width, "sd": shelf_depth})
        session.flush()
        rid = session.execute(text("SELECT id FROM racks WHERE name=:name"), {"name": rname}).scalar()
        gc, row, col = 10, (rn-1)//10, (rn-1)%10
        bx, by = (zone_id-1)*100+col*10, row*5
        for shelf in range(1, num_shelves+1):
            for pos in range(1, pps+1):
                session.execute(text("""
                    INSERT INTO slots (rack_id,shelf_number,position_on_shelf,width_cm,height_cm,depth_cm,
                                       current_weight_kg,max_weight_kg,x_coord,y_coord)
                    VALUES (:rid,:sn,:pos,:w,:h,:d,0.0,:mw,:x,:y)
                """), {"rid": rid, "sn": shelf, "pos": pos, "w": round(shelf_width/pps,1),
                       "h": shelf_height, "d": shelf_depth, "mw": wps,
                       "x": bx+pos*2.0, "y": by+shelf*1.5})
    session.commit()
    return count


def auto_generate_zone_configs(n_items: int, total_racks: int) -> list[dict]:
    """Auto-generate zone configs based on the product mix.

    Analyzes what types of products will be generated and creates zones
    proportionally so every product has a compatible zone with enough space.
    """
    from .test_data import AFFINITY_GROUPS

    # Count expected items per tag category
    # Affinity group items (fixed ~135 items)
    tag_counts: dict[str, int] = {}
    for gn, gr in AFFINITY_GROUPS.items():
        key = ",".join(sorted(gr["tags"]))
        tag_counts[key] = tag_counts.get(key, 0) + len(gr["items"])

    # Filler categories (distributed equally among 7 categories)
    filler_categories = [
        ("grocery", ["grocery"]),
        ("chemical,household", ["chemical", "household"]),
        ("electronics,fragile", ["electronics", "fragile"]),
        ("clothing", ["clothing"]),
        ("general", ["general"]),
        ("heavy,industrial", ["heavy", "industrial"]),
        ("grocery,beverage", ["grocery", "beverage"]),
    ]
    remaining = n_items - sum(tag_counts.values())
    per_cat = max(1, remaining // len(filler_categories))
    for cat_key, tags in filler_categories:
        key = ",".join(sorted(tags))
        tag_counts[key] = tag_counts.get(key, 0) + per_cat

    # Convert tag counts to zone configs
    # Safety tags get dedicated zones, others can share
    total_items = sum(tag_counts.values())
    zone_configs = []
    dist = 5.0  # start 5m from picking, increase for each zone

    # Zone naming map
    tag_zone_names = {
        "grocery": "Grocery",
        "beverage,grocery": "Beverages",
        "grocery,perishable,refrigerated": "Dairy & Frozen",
        "grocery,perishable": "Fresh & Perishable",
        "chemical,household": "Household & Cleaning",
        "flammable,hazardous": "Flammable Storage",
        "electronics,fragile": "Electronics",
        "clothing": "Clothing & Apparel",
        "heavy,industrial": "Tools & Hardware",
        "general": "General Merchandise",
    }

    for tag_key, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        tags = tag_key.split(",")
        name = tag_zone_names.get(tag_key, f"Zone-{tag_key.replace(',','-').title()}")

        # Allocate racks proportional to item count
        pct = count / total_items
        racks = max(2, round(total_racks * pct))

        # Safety-tagged zones go farthest from picking
        is_safety = any(t in ("flammable", "hazardous", "chemical") for t in tags)
        if is_safety:
            zone_dist = 30.0 + dist * 0.5
        else:
            zone_dist = dist
            dist += max(3.0, 15.0 * pct)  # closer zones for bigger categories

        zone_configs.append({
            "name": name, "tags": tags, "distance": round(zone_dist, 1),
            "racks": racks, "shelves_per_rack": 6, "shelf_width": 100.0,
        })

    # Also add a "General Overflow" zone that accepts ANYTHING
    zone_configs.append({
        "name": "General Overflow",
        "tags": ["general"],
        "distance": round(dist + 5, 1),
        "racks": max(2, total_racks // 10),
        "shelves_per_rack": 6,
        "shelf_width": 100.0,
    })

    return zone_configs


def generate_fresh_warehouse(session: Session, zone_configs: list[dict],
                             n_items: int, n_orders: int) -> dict:
    """Generate a complete warehouse from user-provided zone configs.

    zone_configs: list of {name, tags, distance, racks, shelves_per_rack, shelf_width}
    """
    from .database import drop_db, init_db, create_db_engine
    from .models import Base

    # Clear existing data
    for tbl in ["order_history", "items", "slots", "racks", "zones"]:
        try:
            session.execute(text(f"DELETE FROM {tbl}"))
        except Exception:
            pass
    session.commit()

    from .test_data import (
        generate_items, generate_order_history,
        _generate_item_dimensions, AFFINITY_GROUPS,
    )
    import random as _r

    # Create zones
    zone_objs = []
    for zc in zone_configs:
        session.execute(text("""
            INSERT INTO zones (name, tags_json, distance_to_picking_area)
            VALUES (:name, :tags, :dist)
        """), {"name": zc["name"], "tags": json.dumps(zc["tags"]), "dist": zc["distance"]})
    session.flush()

    zones = session.execute(text("SELECT id, name FROM zones ORDER BY id")).fetchall()
    zone_map = {z[1]: z[0] for z in zones}

    total_slots = 0
    for zc in zone_configs:
        zid = zone_map[zc["name"]]
        n_racks = zc.get("racks", 10)
        n_sh = zc.get("shelves_per_rack", 6)
        sw = zc.get("shelf_width", 100.0)
        pps = max(1, int(sw/25))
        wps = round(200.0/n_sh/pps, 2)
        for ri in range(n_racks):
            rname = f"{zc['name']}-R{ri+1:03d}"
            session.execute(text("""
                INSERT INTO racks (zone_id,name,max_weight_kg,num_shelves,
                                   shelf_height_cm,shelf_width_cm,shelf_depth_cm)
                VALUES (:zid,:name,200.0,:ns,40.0,:sw,50.0)
            """), {"zid": zid, "name": rname, "ns": n_sh, "sw": sw})
            session.flush()
            rid = session.execute(text("SELECT id FROM racks WHERE name=:n"), {"n": rname}).scalar()
            row, col = ri//10, ri%10
            bx, by = (zid-1)*100+col*10, row*5
            for sh in range(1, n_sh+1):
                for pos in range(1, pps+1):
                    session.execute(text("""
                        INSERT INTO slots (rack_id,shelf_number,position_on_shelf,
                                           width_cm,height_cm,depth_cm,
                                           current_weight_kg,max_weight_kg,x_coord,y_coord)
                        VALUES (:rid,:sn,:pos,:w,40.0,50.0,0.0,:mw,:x,:y)
                    """), {"rid": rid, "sn": sh, "pos": pos, "w": round(sw/pps,1),
                           "mw": wps, "x": bx+pos*2.0, "y": by+sh*1.5})
                    total_slots += 1
    session.commit()

    # Generate items using existing test_data module (re-import to use ORM)
    from .models import Item, OrderHistory
    items = []
    sku_counter = 0
    filler_categories = [
        ("grocery", ["grocery"]), ("household", ["chemical","household"]),
        ("electronics", ["electronics","fragile"]), ("clothing", ["clothing"]),
        ("general", ["general"]), ("heavy", ["heavy","industrial"]),
        ("beverage", ["grocery","beverage"]),
    ]
    for gn, gr in AFFINITY_GROUPS.items():
        for iname in gr["items"]:
            sku_counter += 1
            dims = _generate_item_dimensions(gr["tags"])
            items.append(Item(sku=f"SKU-{sku_counter:06d}", name=iname,
                              tags_json=json.dumps(gr["tags"]), **dims))
    from .test_data import _generate_product_name
    remaining = n_items - len(items)
    for _ in range(max(0, remaining)):
        sku_counter += 1
        cat_name, tags = _r.choice(filler_categories)
        dims = _generate_item_dimensions(tags)
        items.append(Item(sku=f"SKU-{sku_counter:06d}",
                          name=_generate_product_name(cat_name),
                          tags_json=json.dumps(tags), **dims))
    _r.shuffle(items)
    for start in range(0, len(items), 5000):
        session.add_all(items[start:start+5000])
        session.flush()

    generate_order_history(session, items, n_orders=n_orders//5, target_records=n_orders)
    session.commit()

    return {"zones": len(zone_configs), "slots": total_slots, "items": len(items), "orders": n_orders}


def search_items(session: Session, query: str) -> tuple[list[dict], set[int]]:
    """Search items by name or SKU (case-insensitive partial match).

    Returns (results_list, matching_slot_ids).
    Each result dict has: item_id, name, sku, velocity_class, weight,
    slot_id, zone_name, rack_name, shelf_number, position, group.
    """
    if not query or len(query.strip()) < 2:
        return [], set()

    q = f"%{query.strip()}%"
    item_group_map = _build_item_group_map()

    rows = session.execute(text("""
        SELECT i.id, i.name, i.sku, i.velocity_class, i.weight_kg,
               i.current_slot_id,
               s.shelf_number, s.position_on_shelf,
               r.name AS rack_name,
               z.name AS zone_name, z.distance_to_picking_area
        FROM items i
        LEFT JOIN slots s ON i.current_slot_id = s.id
        LEFT JOIN racks r ON s.rack_id = r.id
        LEFT JOIN zones z ON r.zone_id = z.id
        WHERE LOWER(i.name) LIKE LOWER(:q) OR LOWER(i.sku) LIKE LOWER(:q)
        ORDER BY i.velocity_score DESC
        LIMIT 200
    """), {"q": q}).fetchall()

    results = []
    slot_ids = set()
    for r in rows:
        sid = r[5]
        if sid is not None:
            slot_ids.add(sid)
        results.append({
            "item_id": r[0], "name": r[1], "sku": r[2],
            "velocity_class": r[3], "weight": r[4],
            "slot_id": sid,
            "shelf": r[6], "position": r[7],
            "rack": r[8] or "—", "zone": r[9] or "—",
            "distance": r[10] or 0,
            "group": item_group_map.get(r[1], ""),
        })

    return results, slot_ids


def get_item_full_detail(session: Session, item_id: int) -> dict:
    """Return comprehensive detail for a single item, including location and co-purchase neighbors."""
    row = session.execute(text("""
        SELECT i.id, i.name, i.sku, i.tags_json, i.weight_kg,
               i.width_cm, i.height_cm, i.depth_cm,
               i.velocity_score, i.velocity_class,
               i.current_slot_id,
               s.shelf_number, s.position_on_shelf, s.max_weight_kg AS slot_capacity,
               r.name AS rack_name, r.num_shelves,
               z.name AS zone_name, z.distance_to_picking_area, z.tags_json AS zone_tags,
               COUNT(DISTINCT oh.order_id) AS order_count
        FROM items i
        LEFT JOIN slots s ON i.current_slot_id = s.id
        LEFT JOIN racks r ON s.rack_id = r.id
        LEFT JOIN zones z ON r.zone_id = z.id
        LEFT JOIN order_history oh ON oh.item_id = i.id
        WHERE i.id = :iid
        GROUP BY i.id
    """), {"iid": item_id}).fetchone()

    if not row:
        return {}

    tags = json.loads(row[3]) if row[3] else []
    zone_tags = json.loads(row[18]) if row[18] else []
    item_group_map = _build_item_group_map()
    grp = item_group_map.get(row[1], "")

    # Find co-purchase neighbors (top 8 items bought together with this one)
    neighbors = session.execute(text("""
        SELECT i2.name, i2.sku, COUNT(DISTINCT oh2.order_id) AS copurchase_count
        FROM order_history oh1
        JOIN order_history oh2 ON oh1.order_id = oh2.order_id AND oh1.item_id != oh2.item_id
        JOIN items i2 ON oh2.item_id = i2.id
        WHERE oh1.item_id = :iid
        GROUP BY i2.id
        ORDER BY copurchase_count DESC
        LIMIT 8
    """), {"iid": item_id}).fetchall()

    return {
        "id": row[0], "name": row[1], "sku": row[2],
        "tags": tags, "weight": row[4],
        "width": row[5], "height": row[6], "depth": row[7],
        "velocity_score": row[8], "velocity_class": row[9],
        "placed": row[10] is not None,
        "shelf": row[11], "position": row[12], "slot_capacity": row[13],
        "rack": row[14] or "—", "num_shelves": row[15],
        "zone": row[16] or "—", "zone_distance": row[17] or 0,
        "zone_tags": zone_tags,
        "order_count": row[19] or 0,
        "group": grp,
        "neighbors": [{"name": n[0], "sku": n[1], "count": n[2]} for n in neighbors],
    }


# ═══════════════════════════════════════════════════════════════
# PATENT FEATURE: SELF-HEALING SLOTTING ENGINE
# ═══════════════════════════════════════════════════════════════

def compute_health_score(session: Session) -> dict:
    """Compute Placement Health Score (0-100).

    Compares each item's current zone distance against its optimal zone
    distance given its velocity class. Returns overall score + per-class breakdown.
    """
    rows = session.execute(text("""
        SELECT i.id, i.velocity_class, i.velocity_score,
               z.distance_to_picking_area AS cur_dist
        FROM items i
        JOIN slots s ON i.current_slot_id = s.id
        JOIN racks r ON s.rack_id = r.id
        JOIN zones z ON r.zone_id = z.id
        WHERE i.current_slot_id IS NOT NULL
    """)).fetchall()

    if not rows:
        return {"score": 0, "total_items": 0, "misplaced": 0, "details": {}}

    zone_dists = session.execute(text(
        "SELECT MIN(distance_to_picking_area), MAX(distance_to_picking_area) FROM zones"
    )).fetchone()
    min_d, max_d = (zone_dists[0] or 5), (zone_dists[1] or 40)
    range_d = max_d - min_d if max_d > min_d else 1

    optimal_ranges = {"A": (min_d, min_d + range_d * 0.3),
                      "B": (min_d + range_d * 0.2, min_d + range_d * 0.7),
                      "C": (min_d + range_d * 0.5, max_d)}

    total_penalty = 0
    misplaced = 0
    class_stats = {"A": {"count": 0, "ok": 0}, "B": {"count": 0, "ok": 0}, "C": {"count": 0, "ok": 0}}

    for r in rows:
        vc = r[1] or "C"
        cur_dist = r[3]
        lo, hi = optimal_ranges.get(vc, (min_d, max_d))
        class_stats.setdefault(vc, {"count": 0, "ok": 0})
        class_stats[vc]["count"] += 1

        if lo <= cur_dist <= hi:
            class_stats[vc]["ok"] += 1
        else:
            deviation = max(0, cur_dist - hi) if cur_dist > hi else max(0, lo - cur_dist)
            total_penalty += deviation / range_d
            misplaced += 1

    n = len(rows)
    score = max(0, min(100, 100 - (total_penalty / max(n, 1)) * 200))

    return {
        "score": round(score, 1), "total_items": n, "misplaced": misplaced,
        "well_placed": n - misplaced, "details": class_stats,
    }


def generate_healing_plan(session: Session, max_moves: int = 20) -> list[dict]:
    """Generate a minimal-disruption healing plan.

    Finds the top-N item swaps/moves that recover the most health score
    with the least labor effort.
    """
    rows = session.execute(text("""
        SELECT i.id, i.name, i.sku, i.velocity_class, i.velocity_score,
               z.name AS cur_zone, z.distance_to_picking_area AS cur_dist,
               r.name AS cur_rack,
               COUNT(DISTINCT oh.order_id) AS order_count
        FROM items i
        JOIN slots s ON i.current_slot_id = s.id
        JOIN racks r ON s.rack_id = r.id
        JOIN zones z ON r.zone_id = z.id
        LEFT JOIN order_history oh ON oh.item_id = i.id
        WHERE i.current_slot_id IS NOT NULL
        GROUP BY i.id
    """)).fetchall()

    zone_dists = session.execute(text(
        "SELECT name, distance_to_picking_area FROM zones ORDER BY distance_to_picking_area"
    )).fetchall()
    if not zone_dists:
        return []

    min_d = zone_dists[0][1]
    max_d = zone_dists[-1][1]
    range_d = max_d - min_d if max_d > min_d else 1

    optimal_target = {"A": zone_dists[0], "B": zone_dists[len(zone_dists)//2], "C": zone_dists[-1]}

    moves = []
    for r in rows:
        vc = r[3] or "C"
        cur_dist = r[6]
        target = optimal_target.get(vc, zone_dists[-1])
        target_dist = target[1]

        health_gain = abs(cur_dist - target_dist) / range_d * 10
        if health_gain < 0.5:
            continue

        daily_picks = (r[8] or 0) / 365.0
        walk_saved = daily_picks * (cur_dist - target_dist) * 2

        moves.append({
            "name": r[1], "sku": r[2], "vc": vc,
            "cur_zone": r[5], "cur_dist": cur_dist,
            "target_zone": target[0], "target_dist": target_dist,
            "health_gain": round(health_gain, 2),
            "walk_saved_m": round(walk_saved, 1),
            "priority": "High" if health_gain > 3 else "Medium" if health_gain > 1 else "Low",
        })

    moves.sort(key=lambda x: x["health_gain"], reverse=True)
    return moves[:max_moves]


# ═══════════════════════════════════════════════════════════════
# PATENT FEATURE: TEMPORAL VELOCITY DECAY + SEASONAL DETECTION
# ═══════════════════════════════════════════════════════════════

def compute_temporal_velocity(session: Session) -> list[dict]:
    """Compute time-decayed velocity scores with exponential weighting.

    Recent orders count more than old ones: weight = exp(-lambda * days_ago).
    Also detects items whose order pattern varies significantly by month (seasonal).
    """
    rows = session.execute(text("""
        SELECT i.id, i.name, i.sku, i.velocity_class,
               oh.ordered_at, COUNT(*) as qty
        FROM items i
        JOIN order_history oh ON oh.item_id = i.id
        WHERE i.current_slot_id IS NOT NULL
        GROUP BY i.id, strftime('%Y-%m', oh.ordered_at)
        ORDER BY i.id
    """)).fetchall()

    if not rows:
        return []

    from datetime import datetime
    now = datetime(2026, 1, 1)
    LAMBDA = 0.01  # half-life ~70 days

    item_data = {}
    for r in rows:
        iid = r[0]
        if iid not in item_data:
            item_data[iid] = {"name": r[1], "sku": r[2], "vc": r[3],
                              "monthly_counts": [], "decayed_score": 0}
        try:
            order_date = datetime.strptime(r[4][:7], "%Y-%m") if isinstance(r[4], str) else r[4]
            days_ago = max(0, (now - order_date).days)
        except (ValueError, TypeError):
            days_ago = 180
        decay_weight = math.exp(-LAMBDA * days_ago)
        item_data[iid]["decayed_score"] += r[5] * decay_weight
        item_data[iid]["monthly_counts"].append(r[5])

    max_score = max((d["decayed_score"] for d in item_data.values()), default=1)

    results = []
    for iid, d in item_data.items():
        norm_score = d["decayed_score"] / max_score if max_score > 0 else 0
        counts = d["monthly_counts"]
        avg_c = sum(counts) / len(counts) if counts else 0
        max_c = max(counts) if counts else 0
        variance_ratio = max_c / avg_c if avg_c > 0 else 1
        is_seasonal = variance_ratio > 2.0 and len(counts) >= 3

        results.append({
            "name": d["name"], "sku": d["sku"],
            "current_class": d["vc"],
            "decayed_score": round(norm_score, 4),
            "flat_score_approx": round(sum(counts) / max(sum(c for dd in item_data.values() for c in dd["monthly_counts"]), 1), 4),
            "is_seasonal": is_seasonal,
            "variance_ratio": round(variance_ratio, 1),
            "peak_month_orders": max_c,
            "avg_month_orders": round(avg_c, 1),
            "predicted_class": "A" if norm_score > 0.6 else "B" if norm_score > 0.3 else "C",
            "class_change": d["vc"] != ("A" if norm_score > 0.6 else "B" if norm_score > 0.3 else "C"),
        })

    results.sort(key=lambda x: x["decayed_score"], reverse=True)
    return results[:100]


# ═══════════════════════════════════════════════════════════════
# PATENT FEATURE: ORDER-PATH SIMULATION (PICK ROUTE)
# ═══════════════════════════════════════════════════════════════

def simulate_pick_path(session: Session, item_names: list[str]) -> dict:
    """Simulate a picker's walking route for a set of items.

    Uses nearest-neighbor TSP heuristic to compute shortest path.
    Returns ordered route with coordinates, total distance, and comparison.
    """
    if not item_names:
        return {"route": [], "total_distance": 0, "random_distance": 0, "savings_pct": 0}

    placeholders = ",".join([f"'{n.strip()}'" for n in item_names if n.strip()])
    if not placeholders:
        return {"route": [], "total_distance": 0, "random_distance": 0, "savings_pct": 0}

    rows = session.execute(text(f"""
        SELECT i.name, i.sku, s.x_coord, s.y_coord,
               z.name AS zone_name, r.name AS rack_name, s.shelf_number
        FROM items i
        JOIN slots s ON i.current_slot_id = s.id
        JOIN racks r ON s.rack_id = r.id
        JOIN zones z ON r.zone_id = z.id
        WHERE i.name IN ({placeholders})
    """)).fetchall()

    if len(rows) < 2:
        route = [{"name": r[0], "sku": r[1], "x": r[2], "y": r[3],
                  "zone": r[4], "rack": r[5], "shelf": r[6]} for r in rows]
        return {"route": route, "total_distance": 0, "random_distance": 0, "savings_pct": 0}

    points = [{"name": r[0], "sku": r[1], "x": r[2], "y": r[3],
               "zone": r[4], "rack": r[5], "shelf": r[6]} for r in rows]

    # Nearest-neighbor TSP heuristic starting from picking area (0,0)
    start = {"name": "PICKING AREA", "sku": "", "x": 0, "y": 0, "zone": "Start", "rack": "", "shelf": 0}
    unvisited = list(range(len(points)))
    route = [start]
    current = start
    total_dist = 0

    while unvisited:
        best_i = min(unvisited, key=lambda i: math.sqrt(
            (current["x"] - points[i]["x"])**2 + (current["y"] - points[i]["y"])**2))
        d = math.sqrt((current["x"] - points[best_i]["x"])**2 +
                      (current["y"] - points[best_i]["y"])**2)
        total_dist += d
        current = points[best_i]
        route.append(current)
        unvisited.remove(best_i)

    # Return to picking area
    d_back = math.sqrt(current["x"]**2 + current["y"]**2)
    total_dist += d_back
    route.append({"name": "RETURN", "sku": "", "x": 0, "y": 0, "zone": "End", "rack": "", "shelf": 0})

    # Random ordering distance for comparison
    import random as _r
    shuffled = list(range(len(points)))
    _r.shuffle(shuffled)
    rand_dist = 0
    cur = start
    for i in shuffled:
        rand_dist += math.sqrt((cur["x"] - points[i]["x"])**2 + (cur["y"] - points[i]["y"])**2)
        cur = points[i]
    rand_dist += math.sqrt(cur["x"]**2 + cur["y"]**2)

    savings = round((1 - total_dist / rand_dist) * 100, 1) if rand_dist > 0 else 0

    return {
        "route": route,
        "total_distance": round(total_dist, 1),
        "random_distance": round(rand_dist, 1),
        "savings_pct": savings,
        "items_found": len(points),
        "items_requested": len(item_names),
    }


# ═══════════════════════════════════════════════════════════════
# INTELLIGENCE ANALYTICS (existing)
# ═══════════════════════════════════════════════════════════════

# Avg walking speed ~1.2 m/s, avg pick takes ~15s, avg picks/day per item = order_count/365
WALK_SPEED_MPS = 1.2
PICK_TIME_SEC = 15
MOVE_TIME_PER_ITEM_MIN = 3.0  # estimated labor to relocate one item


def get_reslotting_recommendations(session: Session) -> list[dict]:
    """Find items that are suboptimally placed and compute ROI of moving them.

    Identifies:
    - Fast movers in far zones (should be near picking)
    - Slow movers in prime slots (wasting close-to-picking space)
    Returns recommendations sorted by daily walking savings.
    """
    rows = session.execute(text("""
        SELECT i.id, i.name, i.sku, i.velocity_class, i.velocity_score,
               z.name AS cur_zone, z.distance_to_picking_area AS cur_dist,
               r.name AS cur_rack, s.shelf_number,
               COUNT(DISTINCT oh.order_id) AS order_count
        FROM items i
        JOIN slots s ON i.current_slot_id = s.id
        JOIN racks r ON s.rack_id = r.id
        JOIN zones z ON r.zone_id = z.id
        LEFT JOIN order_history oh ON oh.item_id = i.id
        WHERE i.current_slot_id IS NOT NULL
        GROUP BY i.id
        HAVING order_count > 0
        ORDER BY i.velocity_score DESC
    """)).fetchall()

    if not rows:
        return []

    zone_dists = session.execute(text(
        "SELECT name, distance_to_picking_area FROM zones ORDER BY distance_to_picking_area"
    )).fetchall()
    closest_zone = zone_dists[0] if zone_dists else ("—", 5.0)
    farthest_zone = zone_dists[-1] if zone_dists else ("—", 40.0)

    recs = []
    for r in rows:
        item_id, name, sku, vc, vscore, cur_zone, cur_dist, cur_rack, shelf, oc = r
        daily_picks = oc / 365.0

        if vc == "A" and cur_dist > 8:
            optimal_dist = closest_zone[1]
            target_zone = closest_zone[0]
        elif vc == "B" and cur_dist > 20:
            mid_idx = len(zone_dists) // 2
            optimal_dist = zone_dists[mid_idx][1]
            target_zone = zone_dists[mid_idx][0]
        elif vc == "C" and cur_dist < 10 and daily_picks < 1.0:
            optimal_dist = farthest_zone[1]
            target_zone = farthest_zone[0]
        else:
            continue

        cur_daily_walk_m = daily_picks * cur_dist * 2  # round trip
        opt_daily_walk_m = daily_picks * optimal_dist * 2
        saved_m = cur_daily_walk_m - opt_daily_walk_m
        saved_min = saved_m / WALK_SPEED_MPS / 60

        move_cost_min = MOVE_TIME_PER_ITEM_MIN
        payback_days = move_cost_min / saved_min if saved_min > 0 else 999

        if abs(saved_m) < 1:
            continue

        recs.append({
            "name": name, "sku": sku, "vc": vc,
            "order_count": oc, "daily_picks": round(daily_picks, 2),
            "cur_zone": cur_zone, "cur_dist": cur_dist,
            "target_zone": target_zone, "target_dist": optimal_dist,
            "daily_walk_saved_m": round(saved_m, 1),
            "daily_time_saved_min": round(saved_min, 2),
            "move_cost_min": move_cost_min,
            "payback_days": round(payback_days, 1),
            "action": "Move closer" if vc == "A" else "Move farther",
        })

    recs.sort(key=lambda x: x["daily_walk_saved_m"], reverse=True)
    return recs[:50]


def get_congestion_analysis(session: Session) -> list[dict]:
    """Compute congestion scores per zone.

    Congestion = too many fast movers crammed in one area, causing picker traffic.
    """
    rows = session.execute(text("""
        SELECT z.id, z.name, z.distance_to_picking_area,
               COUNT(DISTINCT r.id) AS rack_count,
               COUNT(DISTINCT s.id) AS slot_count,
               SUM(CASE WHEN i.velocity_class='A' THEN 1 ELSE 0 END) AS a_count,
               SUM(CASE WHEN i.velocity_class='B' THEN 1 ELSE 0 END) AS b_count,
               SUM(CASE WHEN i.id IS NOT NULL THEN 1 ELSE 0 END) AS total_items
        FROM zones z
        LEFT JOIN racks r ON r.zone_id = z.id
        LEFT JOIN slots s ON s.rack_id = r.id
        LEFT JOIN items i ON i.current_slot_id = s.id
        GROUP BY z.id ORDER BY z.distance_to_picking_area
    """)).fetchall()

    results = []
    for r in rows:
        zid, zname, dist, rc, sc, ac, bc, ti = r
        ac = ac or 0; bc = bc or 0; ti = ti or 0; rc = rc or 1; sc = sc or 1

        a_density = ac / sc * 100 if sc > 0 else 0
        items_per_rack = ti / rc if rc > 0 else 0
        # Congestion = high A-density means many pickers converge on same area
        congestion_score = min(100, a_density * 1.5 + (items_per_rack / 50 * 20))

        status = "Critical" if congestion_score > 70 else "Warning" if congestion_score > 40 else "OK"

        results.append({
            "zone": zname, "distance": dist,
            "racks": rc, "slots": sc,
            "fast_movers": ac, "regular": bc,
            "total_items": ti,
            "a_density_pct": round(a_density, 1),
            "items_per_rack": round(items_per_rack, 1),
            "congestion_score": round(congestion_score, 1),
            "status": status,
        })
    return results


def get_cross_category_pairs(session: Session) -> list[dict]:
    """Find co-purchased items from different tag categories placed in the same zone.

    Relaxed from same-rack to same-zone for smaller datasets. Uses a simpler
    two-step approach instead of a massive self-join.
    """
    # Step 1: Find frequently co-purchased item pairs
    copurchase_rows = session.execute(text("""
        SELECT oh1.item_id AS id1, oh2.item_id AS id2, COUNT(DISTINCT oh1.order_id) AS cnt
        FROM order_history oh1
        JOIN order_history oh2 ON oh1.order_id = oh2.order_id AND oh1.item_id < oh2.item_id
        GROUP BY oh1.item_id, oh2.item_id
        HAVING cnt >= 2
        ORDER BY cnt DESC
        LIMIT 500
    """)).fetchall()

    if not copurchase_rows:
        return []

    # Step 2: For top pairs, check if they're placed nearby and have different tags
    results = []
    for cp in copurchase_rows:
        id1, id2, cnt = cp
        pair_info = session.execute(text("""
            SELECT i1.name, i1.tags_json, i1.sku,
                   i2.name, i2.tags_json, i2.sku,
                   r1.name, z1.name,
                   r2.name, z2.name,
                   ABS(s1.x_coord - s2.x_coord) + ABS(s1.y_coord - s2.y_coord) AS dist
            FROM items i1
            JOIN items i2 ON i2.id = :id2
            JOIN slots s1 ON i1.current_slot_id = s1.id
            JOIN slots s2 ON i2.current_slot_id = s2.id
            JOIN racks r1 ON s1.rack_id = r1.id
            JOIN racks r2 ON s2.rack_id = r2.id
            JOIN zones z1 ON r1.zone_id = z1.id
            JOIN zones z2 ON r2.zone_id = z2.id
            WHERE i1.id = :id1
        """), {"id1": id1, "id2": id2}).fetchone()

        if not pair_info:
            continue

        tags1 = json.loads(pair_info[1]) if pair_info[1] else []
        tags2 = json.loads(pair_info[4]) if pair_info[4] else []

        if set(tags1) == set(tags2):
            continue

        results.append({
            "item1": pair_info[0], "tags1": tags1, "sku1": pair_info[2],
            "item2": pair_info[3], "tags2": tags2, "sku2": pair_info[5],
            "rack": f"{pair_info[6]} / {pair_info[8]}",
            "zone": f"{pair_info[7]}" + (f" / {pair_info[9]}" if pair_info[7] != pair_info[9] else ""),
            "slot_distance": round(pair_info[10], 1) if pair_info[10] else 0,
            "copurchase_count": cnt,
        })

        if len(results) >= 30:
            break

    return results

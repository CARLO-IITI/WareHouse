"""Streamlit dashboard -- Stadium/BookMyShow-style warehouse visualization.

Run with: streamlit run warehouse_management/dashboard.py
"""
from __future__ import annotations
import json, os, random, sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go_module
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import text as sa_text
from warehouse_management.database import DEFAULT_DB_PATH, create_db_engine, get_session_factory, init_db
from warehouse_management.queries import (
    AFFINITY_GROUP_COLORS,
    add_items_to_db, add_racks_to_db, auto_generate_zone_configs,
    compute_health_score, compute_temporal_velocity,
    generate_fresh_warehouse, generate_healing_plan,
    get_affinity_group_placement, get_congestion_analysis,
    get_cross_category_pairs, get_distance_by_class, get_kpi_metrics,
    get_rack_detail, get_rack_info, get_racks_for_zone,
    get_reslotting_recommendations, get_top_items,
    get_velocity_distribution, get_warehouse_compact_layout,
    get_zone_item_distribution, get_zone_summary, get_zones_list,
    get_item_full_detail, search_items, simulate_pick_path,
)
from warehouse_management.warehouse_graph import (
    build_warehouse_graph, get_graph_stats, get_graph_for_visualization,
    slot_to_picking_distance, slot_to_slot_distance,
)

st.set_page_config(page_title="Warehouse Slotting", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

TAG_COLORS = {
    "grocery":"#22c55e","perishable":"#14b8a6","refrigerated":"#06b6d4",
    "beverage":"#3b82f6","chemical":"#a855f7","household":"#8b5cf6",
    "flammable":"#ef4444","hazardous":"#dc2626","electronics":"#6366f1",
    "fragile":"#818cf8","clothing":"#ec4899","heavy":"#f97316",
    "industrial":"#fb923c","general":"#eab308",
}
ZONE_COLORS = ["#22c55e","#16a34a","#06b6d4","#3b82f6","#a855f7",
               "#ef4444","#6366f1","#ec4899","#f97316","#eab308","#14b8a6","#8b5cf6"]

CSS = """<style>
.stApp{background:#080c18}
.block-container{max-width:1500px;padding-top:1rem}

/* ── PICKING AREA ─── */
.pick-banner{background:linear-gradient(90deg,#065f46,#047857 40%,#10b981);border:2px solid #34d399;
  border-radius:10px;padding:10px 20px;text-align:center;margin-bottom:10px;box-shadow:0 0 24px rgba(16,185,129,.12)}
.pick-banner .t{color:#ecfdf5;font-size:1rem;font-weight:700;letter-spacing:2px}
.pick-banner .s{color:#a7f3d0;font-size:.72rem}

/* ── LEGEND ─── */
.legend{display:flex;align-items:center;gap:14px;padding:6px 14px;background:#0c1020;border:1px solid #1e293b;
  border-radius:8px;margin-bottom:8px;flex-wrap:wrap}
.legend .li{display:flex;align-items:center;gap:4px;font-size:.72rem;color:#cbd5e1}
.legend .ld{width:12px;height:12px;border-radius:2px;display:inline-block}

/* ── ZONE (compact) ─── */
.zc{background:#0c1020;border:1px solid #1a2236;border-radius:8px;margin-bottom:6px;overflow:visible}
.zh{padding:6px 12px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:4px;
  border-bottom:1px solid #141c30}
.zh .zt{font-size:.82rem;font-weight:700;color:#f1f5f9}
.zh .zs{font-size:.65rem;color:#64748b}
.zh .zs b{color:#94a3b8}
.tp{display:inline-block;padding:0 6px;border-radius:8px;font-size:.58rem;font-weight:600;color:#fff;margin:0 1px}
.dt{display:inline-block;background:#141c30;border:1px solid #1e293b;border-radius:5px;
  padding:1px 7px;color:#94a3b8;font-size:.62rem;font-weight:600;margin:0 3px}

/* ── BookMyShow-style warehouse layout ─── */
.bms-wrap{max-width:1200px;margin:0 auto;text-align:center}
.bms-screen{
  background:linear-gradient(90deg,#065f46,#059669,#065f46);
  color:#a7f3d0;font-size:.8rem;font-weight:700;letter-spacing:3px;
  padding:10px 0;margin:0 40px 6px;
  border-radius:0 0 50% 50% / 0 0 18px 18px;
  text-align:center;border:1px solid #34d399;border-top:none;
}
.bms-areas{display:flex;margin:0 40px 12px;gap:2px}
.bms-areas div{flex:1;text-align:center;padding:5px 4px;font-size:.58rem;font-weight:600;
  color:#cbd5e1;border-radius:4px;letter-spacing:.5px}
.bms-section{
  display:flex;align-items:center;gap:8px;
  margin:14px 0 6px;padding:0 4px;
}
.bms-section::before,.bms-section::after{content:'';flex:1;height:1px;background:#1e293b}
.bms-section-txt{white-space:nowrap;font-size:.72rem;font-weight:700;letter-spacing:.5px}
.bms-row{display:flex;align-items:center;gap:0;margin:1px 0;justify-content:center}
.bms-lbl{width:52px;text-align:right;padding-right:8px;font-size:.56rem;color:#475569;
  font-weight:600;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bms-seats{display:inline-flex;gap:2px;flex-wrap:nowrap}
.bms-s{
  width:16px;height:16px;border-radius:3px;display:inline-block;
  cursor:pointer;transition:all .12s;
}
.bms-s:hover{transform:scale(1.6);z-index:50;box-shadow:0 0 8px rgba(255,255,255,.4)}
.bms-fast{background:#4ade80}
.bms-reg{background:#facc15}
.bms-slow{background:#f87171}
.bms-empty{background:#1e293b;border:1px solid #283450}
.bms-search{background:#22d3ee;animation:search-pulse 1.2s ease-in-out infinite}
.bms-dim{opacity:.15}
.bms-gap{width:8px;display:inline-block}
.bms-wall{text-align:center;color:#334155;font-size:.6rem;letter-spacing:2px;
  padding:8px;border-top:2px solid #1e293b;margin-top:10px}

@keyframes search-pulse{0%,100%{box-shadow:0 0 4px #22d3ee}50%{box-shadow:0 0 14px #22d3ee,0 0 24px #06b6d480}}
.srch-results{background:#0c1020;border:1px solid #1a2236;border-radius:8px;padding:10px 14px;margin-top:8px}
.srch-results table{width:100%;border-collapse:collapse;font-size:.72rem;color:#cbd5e1}
.srch-results th{text-align:left;color:#64748b;padding:4px 8px;border-bottom:1px solid #1e293b;font-weight:600}
.srch-results td{padding:4px 8px;border-bottom:1px solid #111827}
.srch-results tr:hover{background:#141c30}
.srch-badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.6rem;font-weight:600;color:#fff}
.grp-legend{display:flex;flex-wrap:wrap;gap:6px;padding:6px 14px;background:#0c1020;
  border:1px solid #1a2236;border-radius:8px;margin-bottom:6px}
.grp-legend .gi{display:flex;align-items:center;gap:3px;font-size:.65rem;color:#94a3b8}
.grp-legend .gd{width:10px;height:10px;border-radius:2px;border:2px solid;display:inline-block}

/* ── METRIC CARD ─── */
.mc{background:linear-gradient(135deg,#1a2236,#0c1020);border:1px solid #243044;
  border-radius:10px;padding:12px;text-align:center}
.mc h3{color:#94a3b8;font-size:.68rem;margin:0 0 4px;text-transform:uppercase;letter-spacing:.4px}
.mc .v{color:#f1f5f9;font-size:1.4rem;font-weight:700}
.mc .u{color:#64748b;font-size:.65rem;margin-top:2px}

/* ── SHELF ROW ─── */
.sr{display:flex;align-items:center;gap:5px;margin-bottom:2px}
.sl{width:55px;font-size:.6rem;color:#64748b;font-weight:600;text-align:right;flex-shrink:0}
.sf{display:flex;gap:2px;flex-wrap:wrap}
.st{width:24px;height:24px;border-radius:3px;display:flex;align-items:center;justify-content:center;
  font-size:.45rem;font-weight:700;cursor:pointer;transition:transform .12s;position:relative;color:#fff}
.st:hover{transform:scale(1.4);z-index:200;box-shadow:0 0 10px rgba(255,255,255,.3)}
.st[data-tip]:hover::after{
  content:attr(data-tip);
  position:absolute;
  bottom:calc(100% + 6px);
  left:50%;
  transform:translateX(-50%) scale(0.714);
  transform-origin:bottom center;
  background:#111827;border:1px solid #4b5563;color:#e5e7eb;
  padding:8px 12px;border-radius:6px;font-size:12px;line-height:1.4;
  white-space:pre-line;z-index:10000;pointer-events:none;
  box-shadow:0 8px 24px rgba(0,0,0,.7);min-width:160px;max-width:260px;text-align:left}
.wb{background:#141c30;border-radius:3px;height:6px;width:100%;margin-top:3px;overflow:hidden}
.wf{height:100%;border-radius:3px}

/* ── AFFINITY ─── */
.ac{background:#0c1020;border:1px solid #1a2236;border-radius:8px;padding:12px;margin-bottom:8px}
.ah{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.at{color:#f1f5f9;font-size:.88rem;font-weight:700}
.as{padding:2px 10px;border-radius:10px;font-size:.7rem;font-weight:700;color:#fff}
.sg2{display:flex;flex-wrap:wrap;gap:3px;margin-top:6px}
.ai{padding:2px 7px;border-radius:5px;font-size:.62rem;font-weight:500}
.ap{background:#1a2e1a;border:1px solid #22c55e;color:#86efac}
.au{background:#141c30;border:1px solid #1e293b;color:#64748b}

/* ── FEATURE CARD ─── */
.fc{background:#0c1020;border:1px solid #1a2236;border-radius:8px;padding:14px;margin-bottom:8px}
.fc h4{color:#38bdf8;margin:0 0 4px;font-size:.85rem}
.fc p{color:#94a3b8;margin:0;font-size:.78rem;line-height:1.3}
</style>"""


def get_db_session():
    p = os.environ.get("WAREHOUSE_DB", DEFAULT_DB_PATH)
    e = create_db_engine(p); init_db(e)
    return get_session_factory(e)()

def mc(label, value, sub=""):
    s = f'<div class="u">{sub}</div>' if sub else ""
    st.markdown(f'<div class="mc"><h3>{label}</h3><div class="v">{value}</div>{s}</div>', unsafe_allow_html=True)

def tp(tags):
    return "".join(f'<span class="tp" style="background:{TAG_COLORS.get(t,"#64748b")}">{t}</span>' for t in tags)

VELOCITY_LABELS = {"A": "Fast Mover", "B": "Regular", "C": "Slow Mover"}
VELOCITY_SHORT  = {"A": "Fast", "B": "Reg", "C": "Slow"}

def vl(vc):
    """Velocity class code → human-readable label."""
    return VELOCITY_LABELS.get(vc, vc or "")

def vs(vc):
    """Velocity class code → short label for tight spaces."""
    return VELOCITY_SHORT.get(vc, vc or "")


def _render_product_detail(detail):
    """Render the product detail panel for the inspector."""
    vc = detail["velocity_class"] or "C"
    vc_colors = {"A": "#4ade80", "B": "#facc15", "C": "#f87171"}
    vc_col = vc_colors.get(vc, "#f87171")

    # Product card
    st.markdown(
        f'<div style="background:#0c1424;border:2px solid {vc_col};border-radius:10px;padding:14px;margin-bottom:6px">'
        f'<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9">{detail["name"]}</div>'
        f'<div style="font-size:.72rem;color:#94a3b8;margin-top:2px">{detail["sku"]}</div>'
        f'<div style="margin-top:6px;display:flex;gap:4px;flex-wrap:wrap">'
        + "".join(f'<span style="background:{TAG_COLORS.get(t,"#64748b")};color:#fff;padding:1px 7px;border-radius:8px;font-size:.58rem;font-weight:600">{t}</span>' for t in detail["tags"])
        + f'</div></div>', unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3 = st.columns(3)
    with c1: mc("Speed", vl(vc))
    with c2: mc("Weight", f"{detail['weight']:.1f}kg")
    with c3: mc("Orders", f"{detail['order_count']:,}")

    # Location
    if detail["placed"]:
        st.markdown(f"📍 **{detail['zone']}** → **{detail['rack']}** → Shelf {detail['shelf']}, Pos {detail['position']}")
        st.caption(f"{detail['zone_distance']}m from picking area · Slot capacity: {detail.get('slot_capacity',0):.0f}kg")
    else:
        st.warning("Not placed on any shelf")

    # Dimensions
    st.caption(f"Dimensions: {detail['width']:.0f} × {detail['height']:.0f} × {detail['depth']:.0f} cm")

    # Co-purchase group
    if detail["group"]:
        st.info(f"🔗 Co-purchase group: **{detail['group']}**")

    # Neighbors
    if detail["neighbors"]:
        st.markdown("**Frequently bought together:**")
        nb_data = [{"Product": n["name"], "SKU": n["sku"], "Times Together": n["count"]}
                   for n in detail["neighbors"][:6]]
        st.dataframe(pd.DataFrame(nb_data), use_container_width=True, hide_index=True,
                     height=min(220, len(nb_data) * 40 + 40))


def _render_prebuild_preview(n_items: int, n_orders: int, zone_configs: list, est_slots: int):
    """Show a preview of what will be generated before building the warehouse."""
    from warehouse_management.test_data import AFFINITY_GROUPS

    # ── A: Product Mix Breakdown ──
    filler_cats = [
        ("Grocery", ["grocery"]),
        ("Household", ["chemical", "household"]),
        ("Electronics", ["electronics", "fragile"]),
        ("Clothing", ["clothing"]),
        ("General", ["general"]),
        ("Tools/Hardware", ["heavy", "industrial"]),
        ("Beverages", ["grocery", "beverage"]),
    ]

    # Count affinity group items per tag set
    aff_count = 0
    aff_by_cat: dict[str, int] = {}
    for gn, gr in AFFINITY_GROUPS.items():
        label = gn.replace("_", " ").title()
        cnt = len(gr["items"])
        aff_count += cnt
        aff_by_cat[label] = cnt

    # Filler distribution
    remaining = max(0, n_items - aff_count)
    per_cat = remaining // len(filler_cats) if filler_cats else 0
    cat_counts = {}
    for cat_name, _ in filler_cats:
        cat_counts[cat_name] = per_cat
    # Add affinity items to their parent categories
    aff_cat_map = {
        "Breakfast": "Grocery", "Pasta Dinner": "Grocery", "Baking": "Grocery",
        "Snacks": "Grocery", "Beverages": "Beverages", "Dairy Cold": "Grocery",
        "Cleaning": "Household", "Personal Care": "Household",
        "Flammable Goods": "Household", "Electronics Acc": "Electronics",
        "Clothing Basics": "Clothing", "Heavy Tools": "Tools/Hardware",
    }
    for aff_name, cnt in aff_by_cat.items():
        parent = aff_cat_map.get(aff_name, "General")
        cat_counts[parent] = cat_counts.get(parent, 0) + cnt

    with st.expander("📊 Preview: What will be generated", expanded=True):
        # Product mix pie chart + metrics
        col_chart, col_info = st.columns([1, 1])

        with col_chart:
            st.markdown("**Product Mix by Category**")
            names = list(cat_counts.keys())
            values = list(cat_counts.values())
            fig = px.pie(names=names, values=values, hole=0.4,
                         color_discrete_sequence=ZONE_COLORS)
            fig.update_layout(height=280, template="plotly_dark", paper_bgcolor="#080c18",
                              margin=dict(l=5, r=5, t=5, b=5), legend=dict(font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown("**Estimated Breakdown**")
            for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
                pct = round(cnt / n_items * 100, 1) if n_items > 0 else 0
                bar_w = min(100, pct * 3)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;font-size:.75rem">'
                    f'<span style="width:90px;color:#cbd5e1;font-weight:600">{cat}</span>'
                    f'<div style="flex:1;height:8px;background:#1e293b;border-radius:3px;overflow:hidden">'
                    f'<div style="width:{bar_w}%;height:100%;background:#3b82f6;border-radius:3px"></div></div>'
                    f'<span style="color:#94a3b8;width:70px;text-align:right">{cnt:,} ({pct}%)</span>'
                    f'</div>', unsafe_allow_html=True)

            # Capacity check
            if n_items > est_slots:
                st.warning(f"⚠️ {n_items:,} products > {est_slots:,} slots — some items won't fit. Add more racks.")
            else:
                st.success(f"✅ {est_slots:,} slots for {n_items:,} products — enough capacity ({round(n_items/est_slots*100)}% fill)")

        # ── B: Affinity Groups ──
        st.markdown("---")
        st.markdown("**🔗 Co-Purchase Affinity Groups** — items the algorithm will try to place NEAR each other")
        cols = st.columns(3)
        for idx, (gn, gr) in enumerate(AFFINITY_GROUPS.items()):
            label = gn.replace("_", " ").title()
            items_str = ", ".join(gr["items"][:6])
            extra = f" +{len(gr['items'])-6} more" if len(gr["items"]) > 6 else ""
            tag_str = ", ".join(gr["tags"])
            with cols[idx % 3]:
                st.markdown(
                    f'<div style="background:#0c1424;border:1px solid #1e293b;border-radius:8px;padding:8px;margin-bottom:6px">'
                    f'<div style="color:#38bdf8;font-weight:700;font-size:.78rem">{label}</div>'
                    f'<div style="color:#64748b;font-size:.58rem;margin:2px 0">{tag_str} · {len(gr["items"])} items</div>'
                    f'<div style="color:#94a3b8;font-size:.68rem">{items_str}{extra}</div>'
                    f'</div>', unsafe_allow_html=True)

        # ── C: Velocity Estimate ──
        st.markdown("---")
        st.markdown("**⚡ Estimated Velocity Distribution** — based on order history")

        # Estimate: ~70% of items will have orders, of those 20%=A, 30%=B, 50%=C
        items_with_orders = min(n_items, int(n_orders * 0.7))
        est_a = int(items_with_orders * 0.20)
        est_b = int(items_with_orders * 0.30)
        est_c = n_items - est_a - est_b

        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1: mc("Fast Movers", f"{est_a:,}", "Near picking area")
        with vc2: mc("Regular", f"{est_b:,}", "Mid-range zones")
        with vc3: mc("Slow Movers", f"{est_c:,}", "Far zones")
        with vc4: mc("Order Records", f"{n_orders:,}", "For affinity analysis")

        st.caption(f"With {n_orders:,} order records, ~{items_with_orders:,} products will have purchase history. "
                   f"Top 20% by order frequency → Fast Movers (placed nearest to picking area).")

        # ── D: Zone-Category Fit ──
        st.markdown("---")
        st.markdown("**🏭 Zone-Category Fit Check** — will every category have enough space?")

        fit_data = []
        for zc in zone_configs:
            zone_slots = zc["racks"] * zc.get("shelves_per_rack", 6) * 4
            # Find categories that match this zone's tags
            matching_cats = []
            for cat_name, cat_tags in filler_cats:
                if set(cat_tags) & set(zc["tags"]) or "general" in zc["tags"]:
                    matching_cats.append(cat_name)
            match_str = ", ".join(matching_cats[:3]) if matching_cats else "Any"
            total_items_for_zone = sum(cat_counts.get(c, 0) for c in matching_cats)

            ok = "✅" if zone_slots >= total_items_for_zone // max(1, len([z for z in zone_configs if set(z["tags"]) & set(zc["tags"])])) else "⚠️"
            fit_data.append({
                "Zone": zc["name"],
                "Tags": ", ".join(zc["tags"]),
                "Racks": zc["racks"],
                "Slots": zone_slots,
                "Matches": match_str,
                "Fit": ok,
            })

        st.dataframe(pd.DataFrame(fit_data), use_container_width=True, hide_index=True, height=min(300, len(fit_data)*40+40))


# ═══════════════════════════════════════════════════════════════
# PAGE 1: WAREHOUSE MAP  (2D floor plan, all zones visible)
# ═══════════════════════════════════════════════════════════════

def _slot_html(sl, highlight_group="", search_slot_ids=None):
    """Render one slot square with correct classes and tooltip.

    Three highlight modes (mutually exclusive, search wins):
    1. search_slot_ids is set → matching slots get pulsing cyan, rest dimmed
    2. highlight_group is set → group items glow in group color, rest dimmed
    3. Neither → normal velocity coloring with affinity group borders
    """
    NL = "&#10;"
    if sl["occ"]:
        vc_cls = {"A": "sa", "B": "sb", "C": "sc"}.get(sl["vc"], "sc")
        grp = sl.get("grp", "")
        grp_label = f"{NL}🔗 Group: {grp}" if grp else ""
        tip = f'{sl["name"]} ({sl["sku"]}){NL}{vl(sl["vc"])} · {sl["w"]:.1f}kg{grp_label}'

        # Mode 1: Search highlight
        if search_slot_ids is not None:
            if sl["id"] in search_slot_ids:
                return f'<div class="ss srch-hl" data-tip="{tip}"></div>'
            return f'<div class="ss {vc_cls} grp-dim" data-tip="{tip}"></div>'

        # Mode 2: Affinity group highlight
        if highlight_group:
            if grp == highlight_group:
                gc = AFFINITY_GROUP_COLORS.get(grp, "#38bdf8")
                return (f'<div class="ss grp-hl" style="background:{gc};border:2px solid {gc};'
                        f'--gc:{gc};box-shadow:0 0 8px {gc}" data-tip="{tip}"></div>')
            return f'<div class="ss {vc_cls} grp-dim" data-tip="{tip}"></div>'

        # Mode 3: Normal with affinity border hints
        if grp:
            gc = AFFINITY_GROUP_COLORS.get(grp, "#38bdf8")
            return f'<div class="ss {vc_cls}" style="border:2px solid {gc}" data-tip="{tip}"></div>'
        return f'<div class="ss {vc_cls}" data-tip="{tip}"></div>'
    else:
        dim = " grp-dim" if (highlight_group or search_slot_ids is not None) else ""
        return f'<div class="ss se{dim}" data-tip="Empty slot"></div>'


def page_warehouse(session):
    st.markdown("### 🗺️ Warehouse Floor Plan")

    result = get_warehouse_compact_layout(session)
    if isinstance(result, tuple):
        layout, group_names = result
    else:
        layout, group_names = result, []

    if not layout:
        st.warning("No warehouse data. Use '⚙️ Manage Warehouse' to create one or run the CLI pipeline.")
        return

    # ── Build the product list FIRST so we know what's selected before rendering the map ──
    all_products = {}       # label -> slot_id
    all_products_iid = {}   # label -> item_id (for detail lookup)
    for zone in layout:
        for rack in zone.get("racks", []):
            for sh_slots in rack["shelves"].values():
                for sl in sh_slots:
                    if sl["occ"]:
                        label = f"{sl['name']}  —  {sl['sku']}  ({vl(sl['vc'])})"
                        all_products[label] = sl["id"]
                        all_products_iid[label] = sl["id"]

    # Legend
    st.markdown(
        '<div class="legend">'
        '<div class="li"><div class="ld" style="background:#4ade80"></div>Fast Mover</div>'
        '<div class="li"><div class="ld" style="background:#facc15"></div>Regular</div>'
        '<div class="li"><div class="ld" style="background:#f87171"></div>Slow Mover</div>'
        '<div class="li"><div class="ld" style="background:#1e293b;border:1px solid #283450"></div>Empty</div>'
        '<div class="li"><div class="ld" style="background:#22d3ee"></div>Selected / Search</div>'
        '<div class="li" style="margin-left:auto;color:#475569;font-size:.6rem">'
        'Hover squares for details · Select product below to highlight on map</div>'
        '</div>', unsafe_allow_html=True)

    # ── Controls row: Search | Product Picker | Group Filter ──
    c_search, c_pick, c_filter = st.columns([2, 2, 2])

    with c_search:
        search_q = st.text_input("🔍 Search", "", key="sku_search",
                                 placeholder="Type product name or SKU...")

    search_results = []
    search_slot_ids = None
    if search_q and len(search_q.strip()) >= 2:
        search_results, search_slot_ids = search_items(session, search_q)

    # Product picker (works without search)
    with c_pick:
        product_list = ["— Pick a product —"] + sorted(all_products.keys())
        selected_product = st.selectbox("📦 Inspect product", product_list, key="product_pick",
                                        label_visibility="visible")

    # Determine the highlight slot from the product picker
    highlight_slot_id = None
    if selected_product != "— Pick a product —":
        highlight_slot_id = all_products.get(selected_product)

    # Group filter (only when no search and no product selected)
    highlight_group = ""
    with c_filter:
        if search_slot_ids:
            st.caption(f"🔍 {len(search_results)} results for \"{search_q}\"")
        elif highlight_slot_id:
            st.caption("📦 Product highlighted on map below")
        else:
            filter_opts = ["Show All"] + [f"🔗 {g}" for g in group_names]
            sel = st.selectbox("🔗 Co-Purchase Group", filter_opts, key="grp_filter")
            highlight_group = "" if sel == "Show All" else sel.replace("🔗 ", "")

    # ── Build BookMyShow-style HTML layout ──
    gt = go_cnt = ga = gb = gc2 = 0
    for z in layout:
        gt += z["total"]; go_cnt += z["occ"]; ga += z["a"]; gb += z["b"]; gc2 += z["c"]

    h = '<div class="bms-wrap">'

    # Screen = Picking Area (like BookMyShow's "SCREEN THIS WAY")
    h += '<div class="bms-screen">🚶 PICKING &amp; DISPATCH AREA</div>'

    # Functional areas bar (Loading / Reception / Outbound)
    h += (
        '<div class="bms-areas">'
        '<div style="background:#1e3a5f">🚛 LOADING DOCKS</div>'
        '<div style="background:#1a3347">📦 RECEPTION</div>'
        '<div style="background:#065f46">🏭 STORAGE ZONES ↓</div>'
        '<div style="background:#1a3347">📤 OUTBOUND</div>'
        '</div>'
    )

    # Each zone = one section, with VISUAL SPACING proportional to distance
    prev_dist = 0
    for idx, zone in enumerate(layout):
        zc = ZONE_COLORS[idx % len(ZONE_COLORS)]
        util = round(zone["occ"]/zone["total"]*100) if zone["total"] else 0

        # Visual gap proportional to distance increase (the further, the more gap)
        dist_gap = zone["distance"] - prev_dist
        gap_px = max(4, min(40, int(dist_gap * 2.5)))
        prev_dist = zone["distance"]
        if idx > 0:
            h += f'<div style="height:{gap_px}px;display:flex;align-items:center;justify-content:center"><span style="color:#334155;font-size:.5rem">↕ {dist_gap:.0f}m walking distance</span></div>'

        # Section header with zone tags and distance bar
        dist_pct = min(100, zone["distance"] / 45 * 100)
        bar_color = "#22c55e" if zone["distance"] <= 10 else "#eab308" if zone["distance"] <= 25 else "#ef4444"
        prox = "NEAR PICKING" if zone["distance"] <= 10 else "MID RANGE" if zone["distance"] <= 25 else "FAR ZONE"

        # Zone tag pills
        tag_pills = ""
        for t in zone.get("tags", []):
            tc = TAG_COLORS.get(t, "#64748b")
            tag_pills += f'<span style="background:{tc};color:#fff;padding:0 5px;border-radius:6px;font-size:.5rem;font-weight:600;margin:0 1px">{t}</span>'

        h += (
            f'<div class="bms-section">'
            f'<span class="bms-section-txt" style="color:{zc}">'
            f'{zone["name"]}'
            f'</span></div>'
            # Zone info row: tags + stats + distance bar
            f'<div style="display:flex;align-items:center;gap:8px;margin:0 30px 4px;flex-wrap:wrap">'
            # Left: tags
            f'<div style="display:flex;gap:2px;align-items:center">{tag_pills}</div>'
            # Center: stats
            f'<span style="font-size:.55rem;color:#64748b">{zone["rack_count"]} racks · {zone["total"]} slots · {util}% full</span>'
            # Right: distance bar
            f'<div style="display:flex;align-items:center;gap:4px;margin-left:auto;min-width:140px">'
            f'<span style="font-size:.52rem;color:#64748b">📍{zone["distance"]}m</span>'
            f'<div style="flex:1;height:4px;background:#1e293b;border-radius:2px;overflow:hidden">'
            f'<div style="width:{dist_pct}%;height:100%;background:{bar_color};border-radius:2px"></div></div>'
            f'<span style="font-size:.5rem;font-weight:700;color:{bar_color}">{prox}</span>'
            f'</div>'
            f'</div>'
        )

        # Each rack = one row (like a seat row A, B, C...)
        for rack in zone.get("racks", []):
            short = rack["name"].split("-")[-1] if "-" in rack["name"] else rack["name"]
            h += f'<div class="bms-row"><span class="bms-lbl">{short}</span><span class="bms-seats">'

            all_sh = sorted(rack["shelves"].keys())
            for si, sh_num in enumerate(all_sh):
                if si > 0:
                    h += '<span class="bms-gap"></span>'

                for sl in rack["shelves"][sh_num]:
                    tip = ""
                    cls = "bms-empty"
                    has_highlight = search_slot_ids or highlight_group or highlight_slot_id

                    if sl["occ"]:
                        grp = sl.get("grp", "")
                        grp_line = f"\n🔗 Group: {grp}" if grp else ""
                        tip = (f'{sl["name"]}\n'
                               f'SKU: {sl["sku"]}\n'
                               f'Speed: {vl(sl["vc"])}\n'
                               f'Weight: {sl["w"]:.1f}kg\n'
                               f'Zone: {zone["name"]} ({zone["distance"]}m)\n'
                               f'Rack: {rack["name"]} | Shelf {sh_num}'
                               f'{grp_line}')

                        # Priority 1: Single product selected from picker
                        if highlight_slot_id and sl["id"] == highlight_slot_id:
                            cls = "bms-search"
                        elif highlight_slot_id and sl["id"] != highlight_slot_id:
                            cls = "bms-empty bms-dim"
                        # Priority 2: Search results
                        elif search_slot_ids and sl["id"] in search_slot_ids:
                            cls = "bms-search"
                        elif search_slot_ids and sl["id"] not in search_slot_ids:
                            cls = "bms-empty bms-dim"
                        # Priority 3: Group highlight
                        elif highlight_group:
                            if grp == highlight_group:
                                cls = "bms-search"
                            else:
                                cls = "bms-empty bms-dim"
                        else:
                            cls = {"A": "bms-fast", "B": "bms-reg", "C": "bms-slow"}.get(sl["vc"], "bms-slow")
                    else:
                        tip = f'Empty slot\nZone: {zone["name"]}\nRack: {rack["name"]} | Shelf {sh_num}'
                        if has_highlight:
                            cls = "bms-empty bms-dim"

                    h += f'<span class="bms-s {cls}" title="{tip}"></span>'

            h += '</span></div>'

    # Bottom wall
    h += '<div class="bms-wall">━━━━━━━━ WAREHOUSE WALL ━━━━━━━━</div>'
    h += '</div>'

    st.markdown(h, unsafe_allow_html=True)

    # ── Product Detail Panel (shows when a product is selected above) ──
    if selected_product != "— Pick a product —" and highlight_slot_id:
        st.markdown("---")
        detail = get_item_full_detail(session, highlight_slot_id)
        if detail:
            _render_product_detail(detail)

    # Summary + Unplaced items explanation
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: mc("Total Slots", f"{gt:,}")
    with c2: mc("Occupied", f"{go_cnt:,}")
    with c3: mc("Utilization", f"{round(go_cnt/gt*100,1) if gt else 0}%")
    with c4: mc("Fast Movers", f"{ga:,}", "Near picking area")
    with c5: mc("Regular + Slow", f"{gb+gc2:,}")

    # Show unplaced items explanation
    kpi = get_kpi_metrics(session)
    unplaced = kpi["unassigned_items"]
    if unplaced > 0:
        pct_unplaced = round(unplaced / kpi["total_items"] * 100, 1) if kpi["total_items"] else 0
        empty_slots = kpi["total_slots"] - kpi["used_slots"]

        if empty_slots == 0:
            reason = "All shelf slots are full — no physical space left."
            fix = "Add more racks in **⚙️ Manage Warehouse → Add Racks**."
        else:
            reason = "Items are too heavy or too large for available slots (even after constraint relaxation)."
            fix = "Add racks with higher weight capacity, or the items will be force-placed on next re-run."

        with st.expander(f"⚠️ {unplaced:,} products not placed ({pct_unplaced}%) — click for details", expanded=False):
            st.markdown(f"**Inventory:** {kpi['total_items']:,} products · **Shelf space:** {kpi['total_slots']:,} slots")
            st.markdown(f"**Filled:** {kpi['used_slots']:,} / {kpi['total_slots']:,} ({kpi['slot_utilization']}%) · **Empty:** {empty_slots}")
            st.markdown(f"**Why:** {reason}")
            st.markdown(f"**Fix:** {fix}")

            # Show actual unplaced items with their specific issue
            unplaced_items = session.execute(sa_text("""
                SELECT i.name, i.tags_json, i.weight_kg, i.width_cm, i.height_cm, i.depth_cm
                FROM items i WHERE i.current_slot_id IS NULL
                ORDER BY i.weight_kg DESC LIMIT 15
            """)).fetchall()
            if unplaced_items:
                slot_max_w = session.execute(sa_text("SELECT MAX(max_weight_kg) FROM slots")).scalar() or 8
                slot_dims = session.execute(sa_text(
                    "SELECT width_cm, height_cm, depth_cm FROM slots LIMIT 1"
                )).fetchone()
                slot_dim_str = f"{slot_dims[0]:.0f}×{slot_dims[1]:.0f}×{slot_dims[2]:.0f}cm" if slot_dims else "?"

                st.markdown(f"**Unplaced items** (slot max weight: {slot_max_w:.1f}kg, slot dims: {slot_dim_str}):")
                for ui in unplaced_items:
                    w = ui[2]
                    dims = sorted([ui[3], ui[4], ui[5]])
                    issues = []
                    if w > slot_max_w * 3:
                        issues.append(f"⚖️ too heavy ({w:.1f}kg vs {slot_max_w:.1f}kg max)")
                    if slot_dims and dims[0] > max(slot_dims):
                        issues.append(f"📐 too large ({dims[0]:.0f}×{dims[1]:.0f}×{dims[2]:.0f}cm)")
                    if not issues:
                        issues.append("❓ unknown constraint")
                    tags = json.loads(ui[1]) if ui[1] else []
                    st.markdown(f"- **{ui[0]}** ({', '.join(tags)}) — {' · '.join(issues)}")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: ZONE EXPLORER (BookMyShow-style rack view)
# ═══════════════════════════════════════════════════════════════
def page_zone_explorer(session):
    st.markdown("### 🔍 Zone & Rack Explorer")
    st.caption("Pick a zone and rack to see every shelf and slot, like zooming into a single theater section")

    zones_df = get_zones_list(session)
    if zones_df.empty:
        st.warning("No warehouse data. Create a warehouse first."); return

    c1, c2 = st.columns(2)
    with c1:
        zo = dict(zip(zones_df["name"], zones_df["id"]))
        sz = st.selectbox("Select Zone", list(zo.keys()), key="ze_z")
        szid = zo[sz]
    rd = get_racks_for_zone(session, szid)
    if rd.empty:
        st.info("This zone has no racks."); return
    with c2:
        ro = dict(zip(rd["name"], rd["id"]))
        sr = st.selectbox("Select Rack", list(ro.keys()), key="ze_r")
        srid = ro[sr]

    ri = get_rack_info(session, srid)
    dt = get_rack_detail(session, srid)
    if dt.empty:
        st.info("This rack has no slots."); return

    # Rack stats
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1: mc("Rack", ri.get("name", "—"))
    with m2: mc("Zone", ri.get("zone_name", "—"))
    with m3: mc("Shelves", str(ri.get("num_shelves", 0)))
    with m4:
        tc = dt["max_weight_kg"].sum(); uw = dt["current_weight_kg"].sum()
        mc("Weight Load", f"{round(uw/tc*100,1) if tc else 0}%", f"{uw:.0f} / {tc:.0f} kg")

    # Legend
    st.markdown(
        '<div class="legend">'
        '<div class="li"><div class="ld" style="background:#4ade80"></div>Fast Mover</div>'
        '<div class="li"><div class="ld" style="background:#facc15"></div>Regular</div>'
        '<div class="li"><div class="ld" style="background:#f87171"></div>Slow Mover</div>'
        '<div class="li"><div class="ld" style="background:#1e293b;border:1px solid #283450"></div>Empty</div>'
        '</div>', unsafe_allow_html=True)

    # Render rack as BookMyShow-style rows: each shelf = one row
    shelves = sorted(dt["shelf_number"].unique())
    mp = int(dt["position_on_shelf"].max())

    h = '<div class="bms-wrap" style="max-width:800px">'
    h += f'<div class="bms-screen" style="margin:0 20px 8px;font-size:.7rem">RACK {ri.get("name","")}</div>'

    for sn in shelves:
        sd = dt[dt["shelf_number"] == sn]
        su = sd["current_weight_kg"].sum()
        sc2 = sd["max_weight_kg"].sum()
        sp = round(su / sc2 * 100) if sc2 else 0
        occ_count = sd["occupied"].sum()
        total_count = len(sd)

        h += f'<div class="bms-row"><span class="bms-lbl">Shelf {sn}</span><span class="bms-seats">'

        for pos in range(1, mp + 1):
            slot = sd[sd["position_on_shelf"] == pos]
            if slot.empty:
                continue
            r = slot.iloc[0]
            if r["occupied"]:
                cls = {"A": "bms-fast", "B": "bms-reg", "C": "bms-slow"}.get(r["velocity_class"], "bms-slow")
                tip = f'{r["item_name"]} ({r["sku"]}) | {vl(r["velocity_class"])} | {r["item_weight"]:.1f}kg'
            else:
                cls = "bms-empty"
                tip = f'Empty | Capacity: {r["max_weight_kg"]:.0f}kg | {r["width_cm"]:.0f}x{r["height_cm"]:.0f}x{r["depth_cm"]:.0f}cm'
            h += f'<span class="bms-s {cls}" title="{tip}"></span>'

        h += '</span>'

        # Weight bar after each shelf
        bc = "#4ade80" if sp < 70 else "#facc15" if sp < 90 else "#f87171"
        h += (f'<span style="width:50px;font-size:.5rem;color:#64748b;text-align:right;padding-left:6px">'
              f'{sp}% wt</span>')
        h += '</div>'

    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)

    # Item list for this rack
    placed = dt[dt["occupied"]]
    if not placed.empty:
        st.markdown("---")
        st.markdown(f"##### Items on this rack ({len(placed)} / {len(dt)} slots)")
        display = placed[["item_name", "sku", "velocity_class", "item_weight", "shelf_number", "position_on_shelf"]].copy()
        display.columns = ["Product", "SKU", "Speed", "Weight (kg)", "Shelf", "Position"]
        display["Speed"] = display["Speed"].map(VELOCITY_LABELS)
        st.dataframe(display, use_container_width=True, hide_index=True, height=min(400, len(display) * 40))


# ═══════════════════════════════════════════════════════════════
# PAGE 3: CO-PURCHASE AFFINITY
# ═══════════════════════════════════════════════════════════════
def page_affinity(session):
    st.markdown("### 🔗 Co-Purchase Affinity Groups")
    st.markdown(
        "These are groups of products that customers frequently buy together "
        "(like bread + butter + milk). The algorithm tries to place them on nearby shelves "
        "so pickers walk less. A high **cluster score** means the group's items are close together."
    )

    groups = get_affinity_group_placement(session)
    if not groups:
        st.warning("No affinity data. Generate a warehouse with order history first."); return

    avg_s = sum(g["cluster_score"] for g in groups) / len(groups)
    wc = sum(1 for g in groups if g["cluster_score"] >= 60)
    tpl = sum(g["placed"] for g in groups)
    tti = sum(g["total"] for g in groups)

    c1, c2, c3, c4 = st.columns(4)
    with c1: mc("Groups", str(len(groups)))
    with c2: mc("Avg Cluster Score", f"{avg_s:.0f}%")
    with c3: mc("Well Clustered", f"{wc}/{len(groups)}", "Score >= 60%")
    with c4: mc("Items Placed", f"{tpl}/{tti}")

    st.markdown("---")

    for g in groups:
        s = g["cluster_score"]
        if s >= 60:
            badge_bg = "#16a34a"; verdict = "Well clustered"
        elif s >= 30:
            badge_bg = "#ca8a04"; verdict = "Partially clustered"
        else:
            badge_bg = "#dc2626"; verdict = "Scattered"

        zs = ", ".join(g["zones_used"]) or "Not placed"

        with st.expander(f"{'🟢' if s>=60 else '🟡' if s>=30 else '🔴'} {g['group']} — {s:.0f}% ({verdict})", expanded=(s < 60)):
            st.markdown(f"**Tags:** {', '.join(g['tags'])}  ·  **Placed:** {g['placed']}/{g['total']}  ·  **Avg distance between items:** {g['avg_distance']}m  ·  **Zones:** {zs}")

            # Items table
            rows = []
            for i in g["items"]:
                rows.append({
                    "Product": i["name"],
                    "Placed": "✅" if i["placed"] else "❌",
                    "Zone": i["zone"] if i["placed"] else "—",
                    "Rack": i.get("rack", "—") if i["placed"] else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=min(300, len(rows) * 40))


# ═══════════════════════════════════════════════════════════════
# Helper: run the slotting pipeline with progress
# ═══════════════════════════════════════════════════════════════
def _run_pipeline(session, clear_first=True):
    """Run the full slotting pipeline with a progress bar."""
    if clear_first:
        session.execute(sa_text("UPDATE items SET current_slot_id=NULL"))
        session.execute(sa_text("UPDATE slots SET current_weight_kg=0.0"))
        session.commit()
    prog = st.progress(0, text="Analyzing order history...")
    try:
        from warehouse_management.slotting_engine import (
            assign_items_to_slots, build_affinity_index,
            compute_velocity_scores, load_warehouse_state, update_velocity_scores_in_db,
        )
        prog.progress(10, text="Scoring items by popularity...")
        vm = compute_velocity_scores(session)
        update_velocity_scores_in_db(session, vm)
        prog.progress(30, text="Finding items bought together (affinity)...")
        ai = build_affinity_index(session, k_neighbors=10)
        prog.progress(60, text="Loading warehouse layout...")
        ws = load_warehouse_state(session)
        prog.progress(70, text="Placing items on shelves...")
        stats = assign_items_to_slots(session, ws, ai, vm, batch_size=1000)
        prog.progress(100, text="Done!")
        st.success(
            f"Placed **{stats['total_assigned']:,}** items on shelves!  \n"
            f"Fast Movers: {stats.get('A',0):,} · Regular: {stats.get('B',0):,} · "
            f"Slow Movers: {stats.get('C',0):,} · Could not place: {stats['total_failed']:,}"
        )
        return stats
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# PAGE 4: MANAGE WAREHOUSE
# ═══════════════════════════════════════════════════════════════
CATEGORY_INFO = {
    "🛒 Grocery":        {"key": "grocery",     "tags": ["grocery"],              "desc": "Rice, pasta, canned goods, snacks, sauces, spices..."},
    "🥛 Dairy & Frozen":  {"key": "beverage",    "tags": ["grocery","perishable","refrigerated"], "desc": "Yogurt, cheese, frozen meals, ice cream..."},
    "🥤 Beverages":       {"key": "beverage",    "tags": ["grocery","beverage"],   "desc": "Water, juice, soda, coffee, tea, energy drinks..."},
    "🧹 Household":       {"key": "household",   "tags": ["chemical","household"], "desc": "Cleaning products, detergent, paper towels..."},
    "📱 Electronics":     {"key": "electronics", "tags": ["electronics","fragile"],"desc": "Cables, chargers, speakers, USB drives..."},
    "👕 Clothing":        {"key": "clothing",    "tags": ["clothing"],             "desc": "T-shirts, pants, socks, shoes, accessories..."},
    "🔧 Tools & Hardware":{"key": "heavy",       "tags": ["heavy","industrial"],   "desc": "Hammers, drills, wrenches, safety gear..."},
    "📦 General":         {"key": "general",     "tags": ["general"],              "desc": "Notebooks, water bottles, yoga mats, pet supplies..."},
}


def page_manage_warehouse(session):
    st.markdown("### ⚙️ Manage Warehouse")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Create New Warehouse",
        "📦 Add Products",
        "🏗️ Add Racks",
        "🔄 Re-Run Placement",
    ])

    # ── TAB 1: Generate full warehouse ──────────────────────────
    with tab1:
        st.markdown("#### Build a new warehouse from scratch")
        st.markdown("This will **replace** your current warehouse with a fresh one.")

        # Step 1: Quick preset
        st.markdown("##### Step 1: Choose a starting size")
        sz = st.radio("", ["Small (demo)", "Medium", "Large", "Custom"], horizontal=True, key="gen_sz", label_visibility="collapsed")

        presets = {
            "Small (demo)": (4, 8, 1000, 5000),
            "Medium":       (6, 20, 10000, 50000),
            "Large":        (10, 50, 50000, 200000),
        }
        default_nz, default_rp, default_ni, default_no = presets.get(sz, (6, 20, 10000, 50000))

        # Step 2: Adjust numbers (always visible)
        st.markdown("##### Step 2: Adjust quantities")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            nz = st.number_input("Zones", 2, 15, default_nz, key="gen_nz3",
                                 help="Separate areas in the warehouse (Grocery, Electronics, etc.)")
        with c2:
            racks_per = st.number_input("Racks per zone", 2, 100, default_rp, key="gen_rp3",
                                        help="Number of shelving units in each zone")
        with c3:
            ni = st.number_input("Products", 100, 200000, default_ni, step=1000, key="gen_ni3",
                                 help="Total products to stock in the warehouse")
        with c4:
            no = st.number_input("Order history", 1000, 1000000, default_no, step=5000, key="gen_no3",
                                 help="Simulated customer orders — more = smarter placement")

        # Step 3: Zone tag assignment
        st.markdown("##### Step 3: Zone tags")
        tag_mode = st.radio("How should zones be set up?", [
            "🤖 Auto-assign (recommended)",
            "✏️ Manual — I'll define zones myself",
        ], key="gen_tag_mode", horizontal=True,
            help="Auto-assign creates zones based on your product mix so every item has a place")

        if tag_mode.startswith("🤖"):
            # Auto-generate zone configs from product mix
            total_racks_est = nz * racks_per
            zone_configs = auto_generate_zone_configs(ni, total_racks_est)

            st.success(f"Auto-generated **{len(zone_configs)} zones** based on {ni:,} products. "
                       f"Each product category gets proportional rack space + a General Overflow zone catches anything that doesn't fit.")

            # Show preview
            with st.expander("📋 Preview auto-generated zones", expanded=False):
                for zc in zone_configs:
                    tag_pills = " ".join(f"`{t}`" for t in zc["tags"])
                    st.markdown(f"- **{zc['name']}** — {tag_pills} — {zc['racks']} racks — {zc['distance']}m from picking")
        else:
            # Manual mode
            preset_zones = [
                ("Grocery", ["grocery"], 5.0),
                ("Dairy & Frozen", ["grocery","perishable","refrigerated"], 10.0),
                ("Beverages", ["grocery","beverage"], 12.0),
                ("Household & Cleaning", ["chemical","household"], 22.0),
                ("Flammable Storage", ["flammable","hazardous"], 38.0),
                ("Electronics", ["electronics","fragile"], 18.0),
                ("Clothing", ["clothing"], 16.0),
                ("Tools & Hardware", ["heavy","industrial"], 32.0),
                ("General Merchandise", ["general"], 8.0),
                ("Pharmacy", ["chemical"], 6.0),
                ("Frozen Foods", ["grocery","perishable","refrigerated"], 14.0),
                ("Stationery", ["general"], 24.0),
                ("Sports & Outdoor", ["general"], 28.0),
                ("Pet Supplies", ["general"], 20.0),
                ("Garden", ["general","heavy"], 35.0),
            ]

            zone_configs = []
            with st.expander(f"📋 Zone Details ({nz} zones) — click to customize", expanded=False):
                st.caption("Each zone has a name, allowed product types (tags), and distance from the picking area.")
                cols = st.columns(3)
                for i in range(nz):
                    p = preset_zones[i] if i < len(preset_zones) else (f"Zone-{i+1}", ["general"], 10.0+i*5)
                    with cols[i % 3]:
                        st.markdown(f"**Zone {i+1}**")
                        name = st.text_input("Name", p[0], key=f"zn2_{i}")
                        tags = st.text_input("Allowed types", ",".join(p[1]), key=f"zt2_{i}")
                        dist = st.number_input("Distance from picking (m)", 1.0, 100.0, p[2], key=f"zd2_{i}")
                        st.markdown("---")
                        zone_configs.append({
                            "name": name, "tags": [t.strip() for t in tags.split(",")],
                            "distance": dist, "racks": racks_per,
                            "shelves_per_rack": 6, "shelf_width": 100.0,
                        })

            if not zone_configs:
                for i in range(nz):
                    p = preset_zones[i] if i < len(preset_zones) else (f"Zone-{i+1}", ["general"], 10.0+i*5)
                    zone_configs.append({
                        "name": p[0], "tags": p[1], "distance": p[2],
                        "racks": racks_per, "shelves_per_rack": 6, "shelf_width": 100.0,
                    })

        total_racks = sum(z["racks"] for z in zone_configs)
        est_slots = sum(z["racks"] * z["shelves_per_rack"] * 4 for z in zone_configs)
        c1,c2,c3,c4 = st.columns(4)
        with c1: mc("Zones", str(len(zone_configs)))
        with c2: mc("Racks", f"{total_racks:,}")
        with c3: mc("~Slots", f"{est_slots:,}")
        with c4: mc("Products", f"{ni:,}")

        # ── Step 4: Preview what will be generated ──
        st.markdown("##### Step 4: Preview")
        _render_prebuild_preview(ni, no, zone_configs, est_slots)

        if st.button("🏗️ Build Warehouse & Place Products", type="primary", key="gen_go2", use_container_width=True):
            with st.spinner("Building warehouse..."):
                result = generate_fresh_warehouse(session, zone_configs, ni, no)
            st.success(f"Built {result['zones']} zones with {result['slots']:,} slots. Added {result['items']:,} products and {result['orders']:,} purchase records.")
            _run_pipeline(session)

    # ── TAB 2: Add products ─────────────────────────────────────
    with tab2:
        st.markdown("#### Add products to your warehouse")
        st.markdown("Products are added to the inventory. Use **Re-Run Placement** (last tab) to assign them to shelves.")

        add_mode = st.radio("", ["Quick Add (pick categories)", "Single Custom Product"], horizontal=True, key="add_m2", label_visibility="collapsed")

        if add_mode == "Quick Add (pick categories)":
            st.markdown("##### Select categories and quantities")
            st.caption("Choose one or more categories. Products with realistic names will be generated with proper tags automatically.")

            selections = {}
            cols = st.columns(2)
            for i, (cat_label, cat_info) in enumerate(CATEGORY_INFO.items()):
                with cols[i % 2]:
                    with st.container():
                        enabled = st.checkbox(cat_label, value=(i < 3), key=f"cat_en_{i}")
                        if enabled:
                            qty = st.slider(f"How many {cat_label.split(' ',1)[1]} products?", 5, 500, 50, key=f"cat_q_{i}")
                            st.caption(cat_info["desc"])
                            selections[cat_label] = (cat_info, qty)

            total_to_add = sum(q for _, q in selections.values())
            if total_to_add > 0:
                st.markdown(f"**Total: {total_to_add} products** across {len(selections)} categories")

            if st.button(f"📦 Add {total_to_add} Products", type="primary", key="add_multi",
                         disabled=total_to_add == 0, use_container_width=True):
                from warehouse_management.test_data import _generate_product_name
                all_data = []
                for cat_label, (cat_info, qty) in selections.items():
                    for _ in range(qty):
                        all_data.append({
                            "name": _generate_product_name(cat_info["key"]),
                            "tags": cat_info["tags"],
                            "weight_kg": round(random.uniform(0.1, 5.0), 2),
                            "width_cm": round(random.uniform(5, 30), 1),
                            "height_cm": round(random.uniform(5, 30), 1),
                            "depth_cm": round(random.uniform(5, 30), 1),
                        })
                added = add_items_to_db(session, all_data)
                st.success(f"Added {added} products! Go to **Re-Run Placement** tab to assign them to shelves.")

        else:
            st.markdown("##### Add a single product with full control")
            with st.form("custom_product"):
                name = st.text_input("Product name", "Organic Almonds 500g")
                cat_sel = st.selectbox("Category", list(CATEGORY_INFO.keys()), key="cp_cat")
                cat_info = CATEGORY_INFO[cat_sel]
                auto_tags = st.checkbox("Use automatic tags for this category", value=True, key="cp_auto")
                if auto_tags:
                    st.info(f"Tags: {', '.join(cat_info['tags'])}")
                    final_tags = cat_info["tags"]
                else:
                    custom_tags = st.text_input("Custom tags (comma separated)", ",".join(cat_info["tags"]), key="cp_tags")
                    final_tags = [t.strip() for t in custom_tags.split(",")]

                c1, c2, c3, c4 = st.columns(4)
                with c1: wt = st.number_input("Weight (kg)", 0.05, 50.0, 1.0, step=0.1)
                with c2: wi = st.number_input("Width (cm)", 1.0, 100.0, 15.0)
                with c3: hi = st.number_input("Height (cm)", 1.0, 100.0, 15.0)
                with c4: di = st.number_input("Depth (cm)", 1.0, 100.0, 15.0)

                if st.form_submit_button("Add Product", type="primary", use_container_width=True):
                    add_items_to_db(session, [{"name": name, "tags": final_tags,
                                               "weight_kg": wt, "width_cm": wi, "height_cm": hi, "depth_cm": di}])
                    st.success(f"Added **{name}**! Go to **Re-Run Placement** to assign it to a shelf.")

    # ── TAB 3: Add racks ────────────────────────────────────────
    with tab3:
        st.markdown("#### Add more racks to an existing zone")
        st.caption("Need more shelf space? Add racks to any zone. Then re-run placement to fill them.")
        zl = get_zones_list(session)
        if zl.empty:
            st.warning("No warehouse exists yet. Create one in the first tab.")
            return
        zo = dict(zip(zl["name"], zl["id"]))
        sel_zone = st.selectbox("Which zone?", list(zo.keys()), key="ar_z2")
        c1, c2 = st.columns(2)
        with c1:
            nr = st.slider("How many racks to add?", 1, 50, 5, key="ar_n2")
            ns = st.slider("Shelves per rack", 2, 12, 6, key="ar_s2")
        with c2:
            sw = st.number_input("Shelf width (cm)", 40.0, 200.0, 100.0, key="ar_w2")
            sh = st.number_input("Shelf height (cm)", 20.0, 80.0, 40.0, key="ar_h2")
            sd = st.number_input("Shelf depth (cm)", 20.0, 80.0, 50.0, key="ar_d2")

        est_new_slots = nr * ns * max(1, int(sw / 25))
        st.markdown(f"This will add **{nr} racks** with ~**{est_new_slots} new slots** to **{sel_zone}**")

        if st.button("🏗️ Add Racks", type="primary", key="ar_btn2", use_container_width=True):
            added = add_racks_to_db(session, zo[sel_zone], nr, ns, sw, sh, sd)
            st.success(f"Added {added} racks ({est_new_slots} slots) to {sel_zone}!")

    # ── TAB 4: Re-run algorithm ─────────────────────────────────
    with tab4:
        st.markdown("#### Re-run the placement algorithm")
        st.markdown(
            "This will **clear all current placements** and re-run the algorithm from scratch.  \n"
            "The algorithm will:\n"
            "1. Score products by popularity (how often they're ordered)\n"
            "2. Find products frequently bought together (e.g. bread + butter)\n"
            "3. Place fast-moving products near the picking area\n"
            "4. Keep co-purchased products on nearby shelves\n"
            "5. Respect zone rules (flammable items only in flammable zone, etc.)"
        )
        kpi = get_kpi_metrics(session)
        c1, c2, c3 = st.columns(3)
        with c1: mc("Products in inventory", f"{kpi['total_items']:,}")
        with c2: mc("Available shelf slots", f"{kpi['total_slots']:,}")
        with c3: mc("Currently placed", f"{kpi['assigned_items']:,}")

        if st.button("🔄 Clear All & Re-Run Placement", type="primary", key="rr2", use_container_width=True):
            _run_pipeline(session)


# ═══════════════════════════════════════════════════════════════
# PAGE 5: KPI DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_kpi(session):
    st.markdown("### 📈 KPI Dashboard")
    m = get_kpi_metrics(session)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: mc("Items",f"{m['total_items']:,}")
    with c2: mc("Assigned",f"{m['assigned_items']:,}")
    with c3: mc("Rate",f"{m['assignment_rate']}%")
    with c4: mc("Slot Util",f"{m['slot_utilization']}%")
    with c5: mc("Orders",f"{m['total_orders']:,}")

    st.markdown("---")
    l,r = st.columns(2)
    with l:
        st.subheader("Items by Movement Speed")
        vd = get_velocity_distribution(session)
        if not vd.empty:
            vd = vd.copy()
            vd["label"] = vd["velocity_class"].map(VELOCITY_LABELS)
            fig = go_module.Figure()
            fig.add_trace(go_module.Bar(x=vd["label"],y=vd["count"],name="Total",marker_color="#3b82f6",
                                 text=vd["count"].apply(lambda x:f"{x:,}"),textposition="outside"))
            fig.add_trace(go_module.Bar(x=vd["label"],y=vd["assigned"],name="Assigned",marker_color="#22c55e",
                                 text=vd["assigned"].apply(lambda x:f"{x:,}"),textposition="outside"))
            fig.update_layout(height=320,template="plotly_dark",paper_bgcolor="#080c18",plot_bgcolor="#080c18",
                              barmode="group",legend=dict(orientation="h",y=1.05),margin=dict(l=30,r=10,t=5,b=30))
            st.plotly_chart(fig, use_container_width=True)
    with r:
        st.subheader("Avg Distance to Picking Area")
        dd = get_distance_by_class(session)
        if not dd.empty:
            dd = dd.copy()
            dd["label"] = dd["velocity_class"].map(VELOCITY_LABELS)
            cs = {"A":"#22c55e","B":"#eab308","C":"#ef4444"}
            fig = go_module.Figure(go_module.Bar(x=dd["label"],y=dd["avg_distance"],
                                   marker_color=[cs.get(c,"#6b7280") for c in dd["velocity_class"]],
                                   text=dd["avg_distance"].apply(lambda x:f"{x}m"),textposition="outside"))
            fig.update_layout(height=320,template="plotly_dark",paper_bgcolor="#080c18",plot_bgcolor="#080c18",
                              yaxis=dict(title="Distance(m)"),margin=dict(l=30,r=10,t=5,b=30))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    l2,r2 = st.columns(2)
    with l2:
        st.subheader("Zone Distribution")
        zd = get_zone_item_distribution(session)
        if not zd.empty:
            fig = px.pie(zd,names="zone_name",values="item_count",hole=.4,
                         color_discrete_sequence=ZONE_COLORS)
            fig.update_layout(height=340,template="plotly_dark",paper_bgcolor="#080c18",
                              margin=dict(l=10,r=10,t=5,b=10),legend=dict(font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)
    with r2:
        st.subheader("Top 10 Items")
        td = get_top_items(session,10)
        if not td.empty:
            td.columns=["SKU","Item","Class","Orders","Zone","Rack","Shelf"]
            st.dataframe(td, use_container_width=True, hide_index=True, height=340)

    st.markdown("---")

    # Additional analytics row
    l3, r3 = st.columns(2)
    with l3:
        st.subheader("Placement Status")
        placed = m["assigned_items"]
        unplaced = m["unassigned_items"]
        fig = px.pie(
            names=["Placed on Shelves", "Not Placed (no space)"],
            values=[placed, unplaced],
            color_discrete_sequence=["#22c55e", "#ef4444"],
            hole=0.45,
        )
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="#080c18",
                          margin=dict(l=10, r=10, t=5, b=10), legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with r3:
        st.subheader("Product Categories")
        cat_rows = session.execute(sa_text("""
            SELECT tags_json, COUNT(*) as cnt FROM items
            GROUP BY tags_json ORDER BY cnt DESC LIMIT 10
        """)).fetchall()
        if cat_rows:
            cat_names = []
            cat_counts = []
            for cr in cat_rows:
                tags = json.loads(cr[0]) if cr[0] else ["unknown"]
                cat_names.append(", ".join(tags))
                cat_counts.append(cr[1])
            fig = px.pie(names=cat_names, values=cat_counts,
                         color_discrete_sequence=ZONE_COLORS, hole=0.4)
            fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="#080c18",
                              margin=dict(l=10, r=10, t=5, b=10), legend=dict(font=dict(size=8)))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Zone Summary")
    zs = get_zone_summary(session)
    if not zs.empty:
        d = zs[["zone_name","distance_to_picking_area","rack_count","slot_count","item_count","utilization_pct"]].copy()
        d.columns=["Zone","Dist(m)","Racks","Slots","Items","Util%"]
        st.dataframe(d, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 6: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
def page_arch(session):
    st.markdown("### 🏗️ Architecture & Features")
    st.markdown("""```
┌──────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD (UI)                       │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│ │ Stadium  │ │  Zone    │ │Co-Purchase│ │Generator │ │  KPI   │ │
│ │  View    │ │Explorer  │ │ Affinity │ │/Simulator│ │Dashboard│ │
│ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└──────┼───────────┼───────────┼───────────┼──────────────┼──────┘
       └───────────┴───────────┴───────────┴──────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                    SLOTTING ENGINE (Python)                       │
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────────┐  │
│ │ Velocity  │ │ Affinity  │ │   Slot    │ │   Constraint     │  │
│ │  Scorer   │ │  Engine   │ │  Matcher  │ │   Enforcer       │  │
│ │(ABC Rank) │ │(BallTree) │ │ (KDTree)  │ │ (Tag/Weight)     │  │
│ └───────────┘ └───────────┘ └───────────┘ └──────────────────┘  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                  DATA LAYER (SQLite + WAL)                        │
│   zones │ racks │ slots │ items │ order_history                   │
└──────────────────────────────────────────────────────────────────┘
```""")
    st.markdown("---")
    st.markdown("""```
Order History ──► Movement Scoring ──► Fast Mover / Regular / Slow Mover
      │                                        │
      └──► Co-Purchase Matrix ──► SVD ──► BallTree KNN
                                                │
              Slot Assignment Engine ◄──────────┘
              ├── Zone/Tag Filter
              ├── Weight/Dimension Check
              ├── KDTree Spatial Search
              └── Multi-Objective Scoring
                    │
              score = 0.5×proximity + 0.3×affinity + 0.2×zone_fit
                    │
              Best Slot Assigned ✓
```""")
    st.markdown("---")
    features = [
        ("⚡","Movement Speed Analysis","Top 20% = Fast Movers (near picking). Next 30% = Regular. Bottom 50% = Slow Movers (far zones). Based on order frequency."),
        ("🔗","BallTree KNN Affinity","Co-purchase matrix → SVD → BallTree O(log n). Bread+butter placed together."),
        ("🏷️","Zone/Tag Enforcement","Flammable→flammable zone only. Chemical→chemical only. Hard constraint."),
        ("⚖️","Weight/Dimension Check","Slot weight limits + physical fit. Vectorized numpy at 100K scale."),
        ("📍","KDTree Spatial Search","scipy KDTree on (x,y) finds slots near already-placed affinity neighbors."),
        ("🎯","Multi-Objective Scoring","Proximity + affinity + zone fit. Fully vectorized."),
        ("🏟️","Stadium View","All zones in ONE view. Green=Fast Mover, Amber=Regular, Red=Slow Mover. Hover for details."),
        ("🔗","Affinity Clustering","See which groups cluster well. Score = how close co-purchased items are."),
        ("🏗️","Warehouse Generator","Define zones, picking areas, racks from UI. Generate + run pipeline."),
        ("💾","SQLite WAL","Write-Ahead Logging. 64MB cache. Batch commits. All FKs indexed."),
    ]
    cols = st.columns(2)
    for i,(ic,ti,de) in enumerate(features):
        with cols[i%2]:
            st.markdown(f'<div class="fc"><h4>{ic} {ti}</h4><p>{de}</p></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 7: INTELLIGENCE (8 tabs — includes 5 patent features)
# ═══════════════════════════════════════════════════════════════
def page_intelligence(session):
    st.markdown("### 🧠 Intelligence & Patent Features")
    st.caption("Self-healing slotting, temporal velocity, pick-path simulation, ergonomic scoring, constraint relaxation")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "❤️ Health Monitor",
        "📈 Temporal Velocity",
        "🛤️ Pick Path Sim",
        "🕸️ Graph View",
        "💰 Re-Slotting ROI",
        "🚧 Congestion",
        "🔗 Cross-Category",
    ])

    # ── TAB 1: Self-Healing Health Monitor ──────────────────────
    with tab1:
        st.markdown("#### ❤️ Placement Health Monitor")
        st.markdown("The system computes a **Health Score (0-100)** by checking if items are in their optimal zones based on current velocity data. When health degrades, it generates a minimal-disruption healing plan.")

        health = compute_health_score(session)
        score = health["score"]
        sc = "#22c55e" if score >= 70 else "#eab308" if score >= 40 else "#ef4444"
        status = "Healthy" if score >= 70 else "Needs Attention" if score >= 40 else "Degraded"

        c1, c2, c3, c4 = st.columns(4)
        with c1: mc("Health Score", f"{score}", f"Status: {status}")
        with c2: mc("Items Analyzed", f"{health['total_items']:,}")
        with c3: mc("Well Placed", f"{health['well_placed']:,}")
        with c4: mc("Misplaced", f"{health['misplaced']:,}")

        # Health gauge bar
        st.markdown(
            f'<div style="background:#141c30;border-radius:8px;height:24px;margin:8px 0;overflow:hidden;position:relative">'
            f'<div style="width:{score}%;height:100%;background:linear-gradient(90deg,{sc},{sc}80);border-radius:8px;'
            f'transition:width 0.5s"></div>'
            f'<div style="position:absolute;top:3px;left:50%;transform:translateX(-50%);color:#fff;font-size:.75rem;font-weight:700">'
            f'{score}% — {status}</div></div>', unsafe_allow_html=True)

        # Class breakdown
        det = health.get("details", {})
        if det:
            st.markdown("##### Per-Class Placement Accuracy")
            cols = st.columns(3)
            for i, vc in enumerate(["A", "B", "C"]):
                d = det.get(vc, {"count": 0, "ok": 0})
                pct = round(d["ok"] / d["count"] * 100) if d["count"] > 0 else 0
                with cols[i]:
                    mc(f"{vl(vc)}", f"{pct}%", f"{d['ok']}/{d['count']} in correct zone")

        # Healing plan
        st.markdown("---")
        st.markdown("##### 🩺 Auto-Generated Healing Plan")
        plan = generate_healing_plan(session, max_moves=15)
        if plan:
            total_health_gain = sum(m["health_gain"] for m in plan)
            st.markdown(f"**{len(plan)} moves** would recover ~**{total_health_gain:.1f}** health points")
            rows_h = ""
            for m in plan:
                pcol = {"High": "#ef4444", "Medium": "#eab308", "Low": "#22c55e"}[m["priority"]]
                rows_h += (
                    f'<tr><td><b>{m["name"]}</b><br><span style="color:#64748b;font-size:.6rem">{m["sku"]}</span></td>'
                    f'<td><span class="srch-badge" style="background:{pcol}">{m["priority"]}</span></td>'
                    f'<td>{m["cur_zone"]} ({m["cur_dist"]}m)</td>'
                    f'<td>→ {m["target_zone"]} ({m["target_dist"]}m)</td>'
                    f'<td style="color:#22c55e;font-weight:700">+{m["health_gain"]}</td>'
                    f'<td>{m["walk_saved_m"]}m/day</td></tr>')
            st.markdown(
                f'<div class="srch-results"><table>'
                f'<tr><th>Product</th><th>Priority</th><th>Current</th><th>Move To</th><th>Health Gain</th><th>Walk Saved</th></tr>'
                f'{rows_h}</table></div>', unsafe_allow_html=True)
        else:
            st.success("No healing moves needed — placement is healthy!")

    # ── TAB 2: Temporal Velocity ─────────────────────────────────
    with tab2:
        st.markdown("#### 📈 Temporal Velocity Decay & Seasonal Detection")
        st.markdown(
            "Standard velocity treats all orders equally. **Temporal decay** weights recent orders "
            "exponentially more (half-life ~70 days). The system also detects **seasonal items** "
            "whose demand varies >2x between months."
        )

        tv_data = compute_temporal_velocity(session)
        if not tv_data:
            st.info("No temporal data available.")
        else:
            seasonal = [d for d in tv_data if d["is_seasonal"]]
            class_changes = [d for d in tv_data if d["class_change"]]

            c1, c2, c3, c4 = st.columns(4)
            with c1: mc("Items Analyzed", str(len(tv_data)))
            with c2: mc("Seasonal Items", str(len(seasonal)))
            with c3: mc("Class Changes", str(len(class_changes)), "Would reclassify with decay")
            with c4:
                pct_change = round(len(class_changes) / len(tv_data) * 100, 1) if tv_data else 0
                mc("Reclassify %", f"{pct_change}%")

            if class_changes:
                st.markdown("---")
                st.markdown("##### 🔄 Items That Would Change Class (with temporal decay)")
                rows_t = ""
                for d in class_changes[:20]:
                    vc_colors = {"A": "#22c55e", "B": "#eab308", "C": "#ef4444"}
                    rows_t += (
                        f'<tr><td><b>{d["name"]}</b><br><span style="color:#64748b;font-size:.6rem">{d["sku"]}</span></td>'
                        f'<td><span class="srch-badge" style="background:{vc_colors.get(d["current_class"],"#64748b")}">{vl(d["current_class"])}</span></td>'
                        f'<td style="font-size:1rem">→</td>'
                        f'<td><span class="srch-badge" style="background:{vc_colors.get(d["predicted_class"],"#64748b")}">{vl(d["predicted_class"])}</span></td>'
                        f'<td>{d["decayed_score"]:.3f}</td>'
                        f'<td>{"🌡️ Seasonal" if d["is_seasonal"] else ""}</td></tr>')
                st.markdown(
                    f'<div class="srch-results"><table>'
                    f'<tr><th>Product</th><th>Current</th><th></th><th>Predicted</th><th>Decay Score</th><th>Flag</th></tr>'
                    f'{rows_t}</table></div>', unsafe_allow_html=True)

            if seasonal:
                st.markdown("---")
                st.markdown("##### 🌡️ Seasonal Items (demand variance >2x)")
                rows_s = ""
                for d in seasonal[:15]:
                    rows_s += (
                        f'<tr><td><b>{d["name"]}</b></td>'
                        f'<td>{d["variance_ratio"]:.1f}x</td>'
                        f'<td>{d["peak_month_orders"]}</td>'
                        f'<td>{d["avg_month_orders"]}</td></tr>')
                st.markdown(
                    f'<div class="srch-results"><table>'
                    f'<tr><th>Product</th><th>Variance</th><th>Peak Month Orders</th><th>Avg Month Orders</th></tr>'
                    f'{rows_s}</table></div>', unsafe_allow_html=True)

    # ── TAB 3: Pick Path Simulation ──────────────────────────────
    with tab3:
        st.markdown("#### 🛤️ Pick Path Simulator")
        st.markdown(
            "Enter product names from a sample order. The system computes the **shortest walking route** "
            "(nearest-neighbor TSP) and compares it against random placement."
        )

        default_items = "White Bread, Butter, Milk 1L, Eggs (12pk), Orange Juice 1L"
        items_input = st.text_area("Products in the order (one per line or comma-separated)",
                                   default_items, height=100, key="pp_items")
        item_names = [n.strip() for n in items_input.replace("\n", ",").split(",") if n.strip()]

        if st.button("🛤️ Simulate Pick Route", type="primary", key="pp_btn") and item_names:
            result = simulate_pick_path(session, item_names)

            if not result["route"]:
                st.warning("No matching items found on shelves.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1: mc("Items Found", f"{result['items_found']}/{result['items_requested']}")
                with c2: mc("Route Distance", f"{result['total_distance']:.0f}m")
                with c3: mc("Random Placement", f"{result['random_distance']:.0f}m")
                with c4:
                    sav = result["savings_pct"]
                    mc("Savings", f"{sav}%", "vs random placement" if sav > 0 else "")

                # Route table
                st.markdown("##### 📍 Optimized Pick Route (in order)")
                rows_p = ""
                for i, stop in enumerate(result["route"]):
                    icon = "🏁" if stop["name"] in ("PICKING AREA", "RETURN") else f"**{i}**"
                    rows_p += (
                        f'<tr><td>{icon}</td><td><b>{stop["name"]}</b></td>'
                        f'<td>{stop.get("zone","")}</td><td>{stop.get("rack","")}</td>'
                        f'<td>({stop["x"]:.0f}, {stop["y"]:.0f})</td></tr>')
                st.markdown(
                    f'<div class="srch-results"><table>'
                    f'<tr><th>#</th><th>Stop</th><th>Zone</th><th>Rack</th><th>Coordinates</th></tr>'
                    f'{rows_p}</table></div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div style="background:#0c1020;border:1px solid #1a2236;border-radius:8px;'
                    f'padding:10px 14px;margin-top:10px;font-size:.75rem;color:#94a3b8">'
                    f'💡 The KNN algorithm placed co-purchased items nearby, reducing the pick route by '
                    f'<b style="color:#22c55e">{sav}%</b> compared to random placement. '
                    f'Total distance: {result["total_distance"]:.0f}m vs {result["random_distance"]:.0f}m random.</div>',
                    unsafe_allow_html=True)

    # ── TAB 4: Graph View ─────────────────────────────────────────
    with tab4:
        st.markdown("#### 🕸️ Warehouse Graph Model")
        st.markdown(
            "The warehouse is modeled as a **NetworkX graph** where nodes are "
            "picking area, zone entries, racks, and slots. Edges are walkable paths "
            "with weights = actual walking distance. The algorithm uses **Dijkstra's shortest path** "
            "instead of naive Euclidean distance."
        )

        try:
            wg = build_warehouse_graph(session)
            stats = get_graph_stats(wg)

            c1, c2, c3, c4 = st.columns(4)
            with c1: mc("Nodes", f"{stats['total_nodes']:,}")
            with c2: mc("Edges", f"{stats['total_edges']:,}")
            with c3: mc("Connected", "Yes" if stats["is_connected"] else "No")
            with c4: mc("Avg Degree", f"{stats['avg_degree']:.1f}")

            st.markdown("**Node types:**")
            for ntype, count in stats["node_types"].items():
                st.markdown(f"- **{ntype}**: {count}")

            # Visualize graph using Plotly
            viz = get_graph_for_visualization(wg)
            if viz["nodes"]:
                node_x = [n["x"] for n in viz["nodes"]]
                node_y = [n["y"] for n in viz["nodes"]]
                node_colors = []
                node_sizes = []
                node_labels = []
                type_colors = {"picking": "#22c55e", "zone": "#3b82f6", "rack": "#eab308", "slot": "#64748b"}
                type_sizes = {"picking": 20, "zone": 14, "rack": 10, "slot": 5}
                for n in viz["nodes"]:
                    node_colors.append(type_colors.get(n["type"], "#64748b"))
                    node_sizes.append(type_sizes.get(n["type"], 5))
                    node_labels.append(f"{n['id']}<br>Type: {n['type']}")

                edge_x, edge_y = [], []
                for e in viz["edges"]:
                    f_node = next((n for n in viz["nodes"] if n["id"] == e["from"]), None)
                    t_node = next((n for n in viz["nodes"] if n["id"] == e["to"]), None)
                    if f_node and t_node:
                        edge_x += [f_node["x"], t_node["x"], None]
                        edge_y += [f_node["y"], t_node["y"], None]

                fig = go_module.Figure()
                fig.add_trace(go_module.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=0.5, color="#334155"),
                    hoverinfo="none", showlegend=False,
                ))
                fig.add_trace(go_module.Scatter(
                    x=node_x, y=node_y, mode="markers",
                    marker=dict(size=node_sizes, color=node_colors, line=dict(width=0)),
                    text=node_labels, hoverinfo="text", showlegend=False,
                ))
                fig.update_layout(
                    height=500, template="plotly_dark",
                    paper_bgcolor="#080c18", plot_bgcolor="#111827",
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    '<div style="background:#0c1020;border:1px solid #1a2236;border-radius:8px;padding:10px;'
                    'font-size:.75rem;color:#94a3b8">'
                    '💡 <b>Green</b> = Picking Area · <b>Blue</b> = Zone Entries · '
                    '<b>Amber</b> = Racks · <b>Grey</b> = Slots. '
                    'Edges represent walkable paths. The algorithm uses Dijkstra shortest-path '
                    'distance through this graph instead of straight-line Euclidean distance.</div>',
                    unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not build graph: {e}")

    # ── TAB 5: Re-Slotting ROI ──────────────────────────────────
    with tab5:
        st.markdown("#### Move Task List — Is it worth relocating these items?")
        st.markdown(
            "The system identifies items placed suboptimally and calculates whether "
            "the labor cost of moving them is justified by the daily walking time saved."
        )

        recs = get_reslotting_recommendations(session)
        if not recs:
            st.info("No re-slotting recommendations. Items appear to be well-placed, or no order history available.")
        else:
            # Summary metrics
            total_walk_saved = sum(r["daily_walk_saved_m"] for r in recs)
            total_time_saved = sum(r["daily_time_saved_min"] for r in recs)
            total_move_cost = sum(r["move_cost_min"] for r in recs)
            avg_payback = sum(r["payback_days"] for r in recs) / len(recs) if recs else 0
            quick_wins = sum(1 for r in recs if r["payback_days"] <= 3)

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: mc("Items to Move", str(len(recs)))
            with c2: mc("Walk Saved/Day", f"{total_walk_saved:.0f}m", f"~{total_walk_saved/1000:.1f}km")
            with c3: mc("Time Saved/Day", f"{total_time_saved:.1f}min")
            with c4: mc("Move Cost", f"{total_move_cost:.0f}min", f"~{total_move_cost/60:.1f}hrs labor")
            with c5: mc("Quick Wins", str(quick_wins), "Payback ≤ 3 days")

            st.markdown("---")

            # Recommendation table
            vc_colors = {"A": "#22c55e", "B": "#eab308", "C": "#ef4444"}
            rows_html = ""
            for r in recs:
                vc_col = vc_colors.get(r["vc"], "#64748b")
                pb_col = "#22c55e" if r["payback_days"] <= 3 else "#eab308" if r["payback_days"] <= 10 else "#ef4444"
                action_icon = "⬆️" if r["action"] == "Move closer" else "⬇️"
                rows_html += (
                    f'<tr>'
                    f'<td><b>{r["name"]}</b><br><span style="color:#64748b;font-size:.6rem">{r["sku"]}</span></td>'
                    f'<td><span class="srch-badge" style="background:{vc_col}">{vl(r["vc"])}</span></td>'
                    f'<td>{r["order_count"]:,} orders<br><span style="color:#64748b;font-size:.6rem">{r["daily_picks"]}/day</span></td>'
                    f'<td>{r["cur_zone"]}<br><span style="color:#64748b;font-size:.6rem">{r["cur_dist"]}m</span></td>'
                    f'<td>{action_icon} {r["target_zone"]}<br><span style="color:#64748b;font-size:.6rem">{r["target_dist"]}m</span></td>'
                    f'<td style="color:#22c55e;font-weight:700">{r["daily_walk_saved_m"]:.0f}m<br>'
                    f'<span style="font-size:.6rem">{r["daily_time_saved_min"]:.1f}min</span></td>'
                    f'<td>{r["move_cost_min"]:.0f}min</td>'
                    f'<td style="color:{pb_col};font-weight:700">{r["payback_days"]}d</td>'
                    f'</tr>'
                )

            st.markdown(
                f'<div class="srch-results">'
                f'<table>'
                f'<tr><th>Product</th><th>Speed</th><th>Orders</th><th>Current</th>'
                f'<th>Move To</th><th>Walk Saved/Day</th><th>Move Cost</th><th>Payback</th></tr>'
                f'{rows_html}'
                f'</table></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="background:#0c1020;border:1px solid #1a2236;border-radius:8px;'
                f'padding:10px 14px;margin-top:10px;font-size:.75rem;color:#94a3b8">'
                f'💡 <b>Interpretation:</b> Moving all {len(recs)} items would cost ~{total_move_cost/60:.1f} hours of labor '
                f'but save {total_walk_saved:.0f}m ({total_walk_saved/1000:.1f}km) of walking <b>every single day</b>. '
                f'Focus on "Quick Wins" (payback ≤ 3 days) first.</div>',
                unsafe_allow_html=True,
            )

    # ── TAB 6: Congestion Analysis ──────────────────────────────
    with tab6:
        st.markdown("#### Aisle Congestion Risk")
        st.markdown(
            "When too many fast-moving items are crammed into the same zone, pickers get stuck "
            "waiting for each other. This shows where congestion is likely."
        )

        cdata = get_congestion_analysis(session)
        if not cdata:
            st.info("No data available.")
        else:
            critical = sum(1 for c in cdata if c["status"] == "Critical")
            warning = sum(1 for c in cdata if c["status"] == "Warning")

            c1, c2, c3 = st.columns(3)
            with c1: mc("Zones Analyzed", str(len(cdata)))
            with c2: mc("Critical", str(critical), "Congestion > 70%")
            with c3: mc("Warning", str(warning), "Congestion 40-70%")

            st.markdown("---")

            for c in cdata:
                score = c["congestion_score"]
                if score > 70:
                    border_col = "#ef4444"; bg = "#1a0a0a"; status_col = "#ef4444"; icon = "🔴"
                elif score > 40:
                    border_col = "#eab308"; bg = "#1a1500"; status_col = "#eab308"; icon = "🟡"
                else:
                    border_col = "#22c55e"; bg = "#0a1a0a"; status_col = "#22c55e"; icon = "🟢"

                bar_w = min(score, 100)
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {border_col}30;border-left:4px solid {border_col};'
                    f'border-radius:8px;padding:10px 14px;margin-bottom:6px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div>'
                    f'<span style="color:#f1f5f9;font-weight:700;font-size:.85rem">{icon} {c["zone"]}</span>'
                    f' <span style="color:#64748b;font-size:.65rem">📍 {c["distance"]}m · {c["racks"]} racks · {c["slots"]} slots</span>'
                    f'</div>'
                    f'<span style="color:{status_col};font-weight:700;font-size:.85rem">{c["status"]} ({score:.0f}%)</span>'
                    f'</div>'
                    f'<div style="display:flex;gap:20px;margin-top:6px;font-size:.7rem;color:#94a3b8">'
                    f'<span>Fast Movers: <b style="color:#22c55e">{c["fast_movers"]}</b></span>'
                    f'<span>Regular: <b style="color:#eab308">{c["regular"]}</b></span>'
                    f'<span>Total Items: <b>{c["total_items"]}</b></span>'
                    f'<span>A-Density: <b>{c["a_density_pct"]}%</b></span>'
                    f'<span>Items/Rack: <b>{c["items_per_rack"]}</b></span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:6px;margin-top:8px;overflow:hidden">'
                    f'<div style="width:{bar_w}%;height:100%;background:{border_col};border-radius:3px"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="background:#0c1020;border:1px solid #1a2236;border-radius:8px;'
                'padding:10px 14px;margin-top:10px;font-size:.75rem;color:#94a3b8">'
                '💡 <b>Fix congestion:</b> Spread fast movers across multiple zones instead of '
                'packing them all near the picking area. Some A-class items can go to mid-distance '
                'zones if it reduces aisle traffic jams.</div>',
                unsafe_allow_html=True,
            )

    # ── TAB 7: Cross-Category Proof ─────────────────────────────
    with tab7:
        st.markdown("#### Cross-Category Clustering Proof")
        st.markdown(
            "These are items from **different product categories** that the KNN algorithm placed "
            "on the **same rack** because customers frequently buy them together. "
            "This proves the algorithm goes beyond simple tag-matching."
        )

        pairs = get_cross_category_pairs(session)
        if not pairs:
            st.info("No cross-category pairs found. This may mean the warehouse is small or items haven't been placed yet.")
        else:
            st.markdown(f"Found **{len(pairs)}** cross-category co-purchase pairs placed nearby:")
            st.markdown("---")

            for p in pairs:
                tags1_html = " ".join(f'<span class="tp" style="background:{TAG_COLORS.get(t,"#64748b")}">{t}</span>' for t in p["tags1"])
                tags2_html = " ".join(f'<span class="tp" style="background:{TAG_COLORS.get(t,"#64748b")}">{t}</span>' for t in p["tags2"])

                co_color = "#22c55e" if p["copurchase_count"] >= 20 else "#eab308" if p["copurchase_count"] >= 10 else "#64748b"

                st.markdown(
                    f'<div style="background:#0c1020;border:1px solid #1a2236;border-radius:8px;'
                    f'padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
                    f'<div style="flex:1;min-width:180px">'
                    f'<div style="color:#f1f5f9;font-weight:600;font-size:.82rem">{p["item1"]}</div>'
                    f'<div style="margin-top:2px">{tags1_html}</div>'
                    f'</div>'
                    f'<div style="color:#475569;font-size:1.2rem">↔</div>'
                    f'<div style="flex:1;min-width:180px">'
                    f'<div style="color:#f1f5f9;font-weight:600;font-size:.82rem">{p["item2"]}</div>'
                    f'<div style="margin-top:2px">{tags2_html}</div>'
                    f'</div>'
                    f'<div style="text-align:right;min-width:140px">'
                    f'<div style="color:{co_color};font-weight:700;font-size:.85rem">🛒 {p["copurchase_count"]}x bought together</div>'
                    f'<div style="color:#64748b;font-size:.65rem">{p["zone"]} · {p["rack"]} · {p["slot_distance"]} slots apart</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="background:#0c1020;border:1px solid #1a2236;border-radius:8px;'
                'padding:10px 14px;margin-top:10px;font-size:.75rem;color:#94a3b8">'
                '💡 <b>What this proves:</b> The BallTree KNN algorithm analyzes actual purchase patterns, '
                'not just product categories. When Pasta (grocery) and Parmesan (dairy) are always bought together, '
                'it places them on the same rack — even though they have different tags. '
                'This minimizes picker walking distance for real customer orders.</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    session = get_db_session()

    st.sidebar.markdown(
        '<div style="text-align:center;padding:8px 0">'
        '<span style="font-size:1.6rem">🏭</span><br>'
        '<span style="color:#f1f5f9;font-size:1rem;font-weight:700">Warehouse Manager</span><br>'
        '<span style="color:#64748b;font-size:.65rem">KNN Slotting System v3</span></div>',
        unsafe_allow_html=True)
    st.sidebar.markdown("---")

    pages = {
        "🗺️ Warehouse Map": page_warehouse,
        "🔍 Zone Explorer": page_zone_explorer,
        "🔗 Affinity Groups": page_affinity,
        "🧠 Intelligence": page_intelligence,
        "⚙️ Manage Warehouse": page_manage_warehouse,
        "📈 KPI Dashboard": page_kpi,
        "🏗️ Architecture": page_arch,
    }
    sel = st.sidebar.radio("Nav", list(pages.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p style="color:#3b4a60;font-size:.6rem;text-align:center">Stadium-Style UI v3</p>',
                        unsafe_allow_html=True)
    pages[sel](session)

if __name__ == "__main__":
    main()
else:
    main()

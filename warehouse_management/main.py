"""CLI entry point for the warehouse slotting optimization system.

Usage:
    python -m warehouse_management.main [--db PATH] [--skip-generate] [--items N]
"""

import argparse
import os
import subprocess
import sys
import time

from .database import (
    create_db_engine,
    drop_db,
    get_session,
    get_session_factory,
    get_table_counts,
    init_db,
)
from .slotting_engine import (
    assign_items_to_slots,
    build_affinity_index,
    compute_assignment_metrics,
    compute_velocity_scores,
    load_warehouse_state,
    update_velocity_scores_in_db,
)
from .test_data import generate_all_test_data


def print_banner():
    print("=" * 70)
    print("  WAREHOUSE SLOTTING OPTIMIZATION SYSTEM")
    print("  KNN-based item placement with zone/tag/weight constraints")
    print("=" * 70)


def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run_pipeline(db_path: str, skip_generate: bool = False, n_items: int = 100_000):
    """Execute the full slotting pipeline."""
    total_start = time.time()
    print_banner()

    # Database setup
    print_section("DATABASE INITIALIZATION")
    engine = create_db_engine(db_path)

    if not skip_generate:
        print("  Dropping existing tables (fresh start)...")
        drop_db(engine)

    init_db(engine)
    session_factory = get_session_factory(engine)
    print(f"  Database: {db_path}")

    with get_session(session_factory) as session:
        # Step 1: Generate test data (or verify existing)
        counts = get_table_counts(session)
        if not skip_generate or counts.get("items", 0) == 0:
            print_section("STEP 1: TEST DATA GENERATION")
            summary = generate_all_test_data(session)
            counts = get_table_counts(session)
        else:
            print_section("STEP 1: USING EXISTING DATA")

        print(f"\n  Table Counts:")
        for table, count in counts.items():
            print(f"    {table:20s}: {count:>10,}")

        # Step 2: Velocity scoring
        print_section("STEP 2: VELOCITY SCORING (ABC Analysis)")
        velocity_map = compute_velocity_scores(session)
        update_velocity_scores_in_db(session, velocity_map)

        # Step 3: Affinity index
        print_section("STEP 3: CO-PURCHASE AFFINITY INDEX (BallTree KNN)")
        affinity_index = build_affinity_index(session, k_neighbors=10)

        # Step 4: Load warehouse state
        print_section("STEP 4: LOADING WAREHOUSE STATE")
        warehouse_state = load_warehouse_state(session)

        # Step 5: Slot assignment
        print_section("STEP 5: SLOT ASSIGNMENT (Multi-objective KDTree)")
        assignment_stats = assign_items_to_slots(
            session,
            warehouse_state,
            affinity_index,
            velocity_map,
            batch_size=1000,
        )

        # Step 6: Metrics
        print_section("STEP 6: ASSIGNMENT RESULTS")
        print(f"\n  Assignment Summary:")
        print(f"    Total items:    {assignment_stats['total_items']:>10,}")
        print(f"    Assigned:       {assignment_stats['total_assigned']:>10,}")
        print(f"    Failed:         {assignment_stats['total_failed']:>10,}")
        print(f"    A-class placed: {assignment_stats.get('A', 0):>10,}")
        print(f"    B-class placed: {assignment_stats.get('B', 0):>10,}")
        print(f"    C-class placed: {assignment_stats.get('C', 0):>10,}")

        if assignment_stats['total_failed'] > 0:
            print(f"\n  Failure Breakdown:")
            print(f"    Tag mismatch:   {assignment_stats.get('failed_tags', 0):>10,}")
            print(f"    Capacity full:  {assignment_stats.get('failed_capacity', 0):>10,}")

        metrics = compute_assignment_metrics(session, warehouse_state)
        print(f"\n  Quality Metrics:")
        print(f"    Assignment rate:   {metrics.get('assignment_rate', 0):>8}%")
        print(f"    Slot utilization:  {metrics.get('slot_utilization', 0):>8}%")

        if "class_A_avg_distance" in metrics:
            print(f"\n  Avg Distance to Picking Area by Velocity Class:")
            for cls in ["A", "B", "C"]:
                key_cnt = f"class_{cls}_count"
                key_dist = f"class_{cls}_avg_distance"
                if key_cnt in metrics:
                    print(f"    Class {cls}: {metrics[key_dist]:>8.1f}m  ({metrics[key_cnt]:,} items)")

    total_elapsed = time.time() - total_start
    print_section("PIPELINE COMPLETE")
    print(f"  Total execution time: {total_elapsed:.1f}s")
    print(f"  Database saved to: {db_path}")
    print()

    return metrics


def launch_dashboard(db_path: str):
    """Launch the Streamlit dashboard."""
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dashboard.py"
    )
    env = os.environ.copy()
    env["WAREHOUSE_DB"] = db_path
    print(f"Launching dashboard (DB: {db_path})...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path,
         "--server.headless", "true"],
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Warehouse Slotting Optimization System",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database file (default: ./warehouse.db)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip test data generation, use existing data",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=100_000,
        help="Number of items to generate (default: 100000)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard UI",
    )

    args = parser.parse_args()

    db_path = args.db or os.path.join(os.getcwd(), "warehouse.db")

    if args.dashboard:
        launch_dashboard(db_path)
        return

    try:
        run_pipeline(db_path, skip_generate=args.skip_generate, n_items=args.items)
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

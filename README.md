# Warehouse Slotting Optimization System

A Python-based fulfilment centre inventory management system that uses KNN-based algorithms to optimally assign 100K+ items to rack/shelf slots. The system respects zone-tag constraints (flammable, chemical, grocery, etc.), weight/dimension limits, co-purchase affinity, and velocity-based proximity to picking areas.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Entry Points                          │
│                     CLI (main.py)                             │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────┐
│                      Core Engine                             │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐                   │
│  │ Velocity Scorer  │  │ Affinity Engine  │                   │
│  │ (ABC Analysis)   │  │ (BallTree KNN)   │                   │
│  └────────┬────────┘  └────────┬─────────┘                   │
│           │                    │                             │
│  ┌────────▼────────────────────▼─────────┐                   │
│  │     Slot Matcher (KDTree spatial)     │                   │
│  │  + Tag/Zone Enforcer                  │                   │
│  │  + Weight/Dimension Validator         │                   │
│  └───────────────────┬───────────────────┘                   │
└──────────────────────┼───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│               Data Layer (SQLite + WAL)                      │
│  zones │ racks │ slots │ items │ order_history                │
└──────────────────────────────────────────────────────────────┘
```

## Algorithms

### 1. Velocity Scoring (ABC Analysis)

Aggregates order history to compute order frequency per item. Items are classified:

- **A-class** (top 20%): Fast movers, placed closest to picking areas
- **B-class** (next 30%): Medium movers
- **C-class** (bottom 50%): Slow movers, placed in remote zones

### 2. Co-Purchase Affinity (BallTree KNN)

1. Builds a sparse item-item co-purchase matrix from order history
2. Reduces dimensionality to 50D via TruncatedSVD (handles the 100K×100K sparsity)
3. Fits a `NearestNeighbors` model with BallTree algorithm for O(log n) neighbor queries
4. When placing an item, prefers slots adjacent to its K nearest co-purchase neighbors

### 3. Slot Assignment (Multi-objective KDTree scoring)

For each item (processed in velocity order, A-class first):

1. **Vectorized constraint filter**: Eliminates slots by zone-tag incompatibility, weight/dimension violations, and occupancy status — all in numpy
2. **KDTree spatial narrowing**: If affinity neighbors are already placed, uses KDTree to find candidate slots near their centroid
3. **Composite scoring**: `score = w1 × proximity + w2 × affinity + w3 × zone_fit`
4. **Assignment**: Best-scoring slot is assigned, in-memory state updated, batched to DB

### 4. Constraint Enforcement

- **Tag/Zone**: Items with restricted tags (flammable, hazardous, chemical, etc.) can ONLY be placed in zones with matching tags
- **Weight**: `slot.current_weight + item.weight ≤ slot.max_weight`
- **Dimensions**: Item dimensions must fit within slot dimensions (checked in any orientation)

## Performance

- **BallTree** gives O(log n) neighbor queries vs O(n) brute force at 100K items
- **KDTree** on slot coordinates enables fast spatial candidate filtering
- **Vectorized numpy operations** for constraint checking (~70-90 items/sec)
- **SQLite WAL mode** with 64MB cache for concurrent read performance
- **Batch DB commits** (2000 items per flush) to minimize I/O overhead
- **Zone compatibility caching** avoids repeated tag set comparisons

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates test data + runs optimization)
python -m warehouse_management.main

# Use an existing database (skip data generation)
python -m warehouse_management.main --skip-generate

# Custom database path
python -m warehouse_management.main --db /path/to/warehouse.db
```

### Requirements

- Python 3.9+
- SQLAlchemy, scikit-learn, scipy, numpy, faker, tqdm

## Test Data

The system generates realistic test data:

| Entity        | Count   | Details                                          |
|---------------|---------|--------------------------------------------------|
| Zones         | 10      | Grocery, Dairy, Chemicals, Flammable, etc.       |
| Racks         | 500     | Distributed proportionally across zones          |
| Slots         | ~14,000 | Varying dimensions and weight capacities         |
| Items         | 100,000 | Tagged by category, realistic sizes/weights      |
| Order History | 500,000 | With co-purchase patterns (bread+butter, etc.)   |

### Co-Purchase Affinity Groups

12 predefined affinity groups simulate real shopping patterns:
- **breakfast**: bread, butter, jam, eggs, milk, cereal
- **pasta_dinner**: pasta, sauce, cheese, olive oil
- **cleaning**: dish soap, detergent, bleach, sponges
- **personal_care**: shampoo, conditioner, toothpaste
- **electronics_acc**: cables, cases, chargers
- And more...

## Project Structure

```
warehouse_management/
├── __init__.py          # Package metadata
├── __main__.py          # python -m entry point
├── models.py            # SQLAlchemy ORM models
├── database.py          # DB engine, sessions, schema
├── test_data.py         # Realistic data generator
├── constraints.py       # Tag/zone/weight/dimension enforcement
├── slotting_engine.py   # Core algorithms (velocity, affinity, assignment)
└── main.py              # CLI pipeline orchestrator
```

## Storage Design

SQLite was chosen for its zero-configuration deployment and excellent single-instance read performance. Key optimizations:

- **WAL journal mode**: Allows concurrent readers during writes
- **64MB page cache**: Reduces disk I/O for repeated queries
- **Indexed columns**: velocity_score, current_slot_id, order_id, item_id
- **Composite spatial index**: (x_coord, y_coord) on slots table

For production multi-user scenarios, the `database.py` module can be swapped to PostgreSQL by changing the connection string — all queries use standard SQL.

## Output Example

```
STEP 6: ASSIGNMENT RESULTS

  Assignment Summary:
    Total items:       100,000
    Assigned:           13,075
    A-class placed:     10,543
    B-class placed:      2,532

  Quality Metrics:
    Assignment rate:      13.1%
    Slot utilization:     94.4%

  Avg Distance to Picking Area by Velocity Class:
    Class A:     15.0m  (10,543 items)
    Class B:     12.9m  (2,532 items)
```

The 94.4% slot utilization shows the system fills nearly all available capacity. The assignment rate is bounded by the physical slot count (~14K slots for 100K items). A-class items are prioritized and fill the majority of slots.

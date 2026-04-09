"""Generate realistic test data for the warehouse slotting system.

Creates 10 zones, 500 racks, ~50K slots, 100K items, and 500K order
history records with realistic co-purchase patterns.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session
import sys


class _NoopProgress:
    """No-op replacement for tqdm when stderr is broken (e.g. inside Streamlit)."""
    def __init__(self, iterable=None, **kw):
        self._it = iterable
    def __iter__(self):
        return iter(self._it) if self._it else iter([])
    def update(self, n=1):
        pass
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


def _safe_tqdm(iterable=None, **kw):
    """tqdm wrapper that silently falls back if stderr is broken."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except (BrokenPipeError, OSError, Exception):
        return _NoopProgress(iterable)

from .models import Item, OrderHistory, Rack, Slot, Zone

ZONE_DEFINITIONS = [
    {"name": "Grocery-A", "tags": ["grocery"], "distance": 5.0},
    {"name": "Grocery-B", "tags": ["grocery", "perishable"], "distance": 8.0},
    {"name": "Dairy-Refrigerated", "tags": ["grocery", "perishable", "refrigerated"], "distance": 12.0},
    {"name": "Beverages", "tags": ["grocery", "beverage"], "distance": 15.0},
    {"name": "Household-Chemicals", "tags": ["chemical", "household"], "distance": 25.0},
    {"name": "Flammable-Storage", "tags": ["flammable", "hazardous"], "distance": 40.0},
    {"name": "Electronics", "tags": ["electronics", "fragile"], "distance": 20.0},
    {"name": "Clothing-Apparel", "tags": ["clothing"], "distance": 18.0},
    {"name": "Heavy-Industrial", "tags": ["heavy", "industrial"], "distance": 35.0},
    {"name": "General-Merchandise", "tags": ["general"], "distance": 10.0},
]

# Affinity groups: items in same group are frequently co-purchased
AFFINITY_GROUPS = {
    "breakfast": {
        "tags": ["grocery"],
        "items": [
            "White Bread", "Whole Wheat Bread", "Butter", "Margarine",
            "Strawberry Jam", "Orange Marmalade", "Peanut Butter",
            "Eggs (12pk)", "Eggs (6pk)", "Milk 1L", "Milk 2L",
            "Corn Flakes", "Oats", "Granola", "Honey",
        ],
    },
    "pasta_dinner": {
        "tags": ["grocery"],
        "items": [
            "Spaghetti", "Penne Pasta", "Fusilli", "Tomato Sauce",
            "Alfredo Sauce", "Parmesan Cheese", "Mozzarella", "Olive Oil",
            "Garlic Cloves", "Basil Leaves", "Ground Beef 500g",
            "Italian Sausage", "Red Wine Vinegar",
        ],
    },
    "baking": {
        "tags": ["grocery"],
        "items": [
            "All-Purpose Flour", "Sugar", "Brown Sugar", "Baking Powder",
            "Baking Soda", "Vanilla Extract", "Cocoa Powder",
            "Chocolate Chips", "Condensed Milk", "Whipping Cream",
        ],
    },
    "snacks": {
        "tags": ["grocery"],
        "items": [
            "Potato Chips", "Tortilla Chips", "Salsa", "Guacamole",
            "Popcorn", "Pretzels", "Trail Mix", "Granola Bars",
            "Cheese Crackers", "Rice Cakes",
        ],
    },
    "beverages": {
        "tags": ["grocery", "beverage"],
        "items": [
            "Cola 2L", "Cola 330ml 6pk", "Orange Juice 1L",
            "Apple Juice 1L", "Sparkling Water 1L", "Green Tea Bags",
            "Black Tea Bags", "Instant Coffee", "Ground Coffee 250g",
            "Energy Drink 4pk",
        ],
    },
    "dairy_cold": {
        "tags": ["grocery", "perishable", "refrigerated"],
        "items": [
            "Greek Yogurt", "Plain Yogurt", "Cheddar Cheese",
            "Cream Cheese", "Sour Cream", "Heavy Cream",
            "Cottage Cheese", "Swiss Cheese", "Butter Unsalted",
            "Gouda Cheese",
        ],
    },
    "cleaning": {
        "tags": ["chemical", "household"],
        "items": [
            "Dish Soap", "Laundry Detergent", "Fabric Softener",
            "Bleach", "Glass Cleaner", "Floor Cleaner",
            "Disinfectant Spray", "Sponges 3pk", "Trash Bags 30pk",
            "Paper Towels",
        ],
    },
    "personal_care": {
        "tags": ["chemical", "household"],
        "items": [
            "Shampoo", "Conditioner", "Body Wash", "Hand Soap",
            "Toothpaste", "Toothbrush", "Deodorant", "Face Wash",
            "Moisturizer", "Sunscreen SPF50",
        ],
    },
    "flammable_goods": {
        "tags": ["flammable", "hazardous"],
        "items": [
            "Lighter Fluid", "Rubbing Alcohol", "Nail Polish Remover",
            "Aerosol Hairspray", "Spray Paint Black", "Spray Paint White",
            "Hand Sanitizer 500ml", "WD-40 Spray", "Butane Refill",
            "Paint Thinner",
        ],
    },
    "electronics_acc": {
        "tags": ["electronics", "fragile"],
        "items": [
            "USB-C Cable", "Lightning Cable", "Phone Case",
            "Screen Protector", "Wireless Earbuds", "Phone Charger",
            "Power Bank 10000mAh", "HDMI Cable", "Mouse Pad",
            "USB Flash Drive 32GB",
        ],
    },
    "clothing_basics": {
        "tags": ["clothing"],
        "items": [
            "T-Shirt White M", "T-Shirt Black M", "Socks 6pk",
            "Boxer Shorts 3pk", "Baseball Cap", "Winter Gloves",
            "Scarf Wool", "Belt Leather", "Hoodie Grey L",
            "Sweatpants Black L",
        ],
    },
    "heavy_tools": {
        "tags": ["heavy", "industrial"],
        "items": [
            "Hammer", "Wrench Set", "Screwdriver Set", "Power Drill",
            "Drill Bits Set", "Duct Tape", "Measuring Tape",
            "Level Tool", "Pliers", "Safety Goggles",
        ],
    },
}

# Realistic product names per category for filler items
PRODUCT_NAMES = {
    "grocery": [
        "Organic Quinoa", "Basmati Rice 1kg", "Rolled Oats 500g", "Jasmine Rice 2kg",
        "Whole Wheat Pasta", "Brown Lentils 500g", "Chickpeas 400g", "Red Kidney Beans",
        "Canned Tomatoes 400g", "Coconut Milk 400ml", "Tomato Ketchup", "Mayonnaise 500g",
        "Mustard Sauce", "Soy Sauce 250ml", "Worcestershire Sauce", "Hot Sauce 150ml",
        "Extra Virgin Olive Oil", "Sunflower Oil 1L", "Sesame Oil 250ml", "Apple Cider Vinegar",
        "Sea Salt 500g", "Black Pepper 100g", "Turmeric Powder 200g", "Cumin Seeds 100g",
        "Cinnamon Sticks", "Paprika 100g", "Oregano Dried 50g", "Bay Leaves 25g",
        "Pancake Mix 500g", "Cake Flour 1kg", "Cornstarch 400g", "Breadcrumbs 300g",
        "Raisins 250g", "Dried Apricots 200g", "Mixed Nuts 300g", "Cashews 250g",
        "Almonds 300g", "Walnuts 200g", "Peanuts Roasted 400g", "Sunflower Seeds 200g",
        "Dark Chocolate Bar", "Milk Chocolate 200g", "White Chocolate 150g", "Gummy Bears 250g",
        "Crackers Whole Wheat", "Rice Crackers 150g", "Digestive Biscuits", "Cookies Choc Chip",
        "Instant Noodles 5pk", "Cup Noodles Chicken", "Ramen Noodles Pack", "Udon Noodles 300g",
        "Canned Tuna 185g", "Canned Sardines 120g", "Canned Corn 340g", "Canned Peas 400g",
        "Strawberry Preserve", "Blueberry Jam 300g", "Mixed Fruit Jam", "Maple Syrup 250ml",
        "Tortilla Wraps 8pk", "Pita Bread 6pk", "Naan Bread 4pk", "Bagels Sesame 6pk",
    ],
    "household": [
        "All-Purpose Cleaner", "Bathroom Cleaner 750ml", "Kitchen Degreaser", "Oven Cleaner 500ml",
        "Window Cleaner 500ml", "Stainless Steel Polish", "Furniture Polish Spray", "Carpet Cleaner",
        "Laundry Pods 42ct", "Fabric Conditioner 1.5L", "Stain Remover Spray", "Color-Safe Bleach",
        "Dryer Sheets 80ct", "Lint Roller Refills", "Wrinkle Release Spray", "Ironing Starch",
        "Dish Soap Lemon 500ml", "Dishwasher Tablets 30ct", "Rinse Aid 250ml", "Dish Brush",
        "Garbage Bags Large 20pk", "Recycling Bags Blue 30pk", "Compost Bags Small 50pk",
        "Paper Towels 6-Roll", "Toilet Paper 12-Roll", "Facial Tissues 3-Box", "Wet Wipes 80ct",
        "Sponge Pack 6ct", "Steel Wool Pads 10ct", "Microfiber Cloths 5pk", "Rubber Gloves M",
        "Air Freshener Lavender", "Scented Candle Vanilla", "Room Spray Ocean", "Reed Diffuser 100ml",
        "Broom & Dustpan Set", "Mop Refill Pads 2pk", "Duster Extendable", "Vacuum Bags 5pk",
        "Moth Balls 250g", "Mouse Trap 2pk", "Ant Killer Spray", "Cockroach Gel Bait",
        "Light Bulb LED 9W", "Light Bulb LED 12W", "Batteries AA 8pk", "Batteries AAA 8pk",
        "Extension Cord 3m", "Power Strip 6-Outlet", "Adhesive Hooks 4pk", "Command Strips 12pk",
    ],
    "electronics": [
        "Bluetooth Speaker Mini", "Wireless Mouse", "USB Keyboard Compact", "Webcam HD 1080p",
        "USB Hub 4-Port", "HDMI Cable 2m", "USB-C Adapter", "Lightning to USB Cable 1m",
        "Wireless Earbuds Pro", "Headphone Stand", "Phone Ring Holder", "Tablet Stand Adjustable",
        "SD Card 64GB", "USB Flash Drive 64GB", "External SSD 256GB", "Micro SD Card 128GB",
        "Screen Cleaning Kit", "Laptop Sleeve 15in", "Keyboard Cover Silicone", "Cable Organizer",
        "Smart Plug WiFi", "LED Strip Lights 5m", "Desk Lamp LED", "Night Light Sensor",
        "Portable Charger 5000mAh", "Car Phone Mount", "Dash Cam Mini", "Action Camera Mount",
        "Surge Protector 8-Outlet", "Travel Adapter Universal", "Wall Charger Dual USB",
        "Wireless Charging Pad", "MagSafe Charger", "Apple Watch Band", "Fitness Tracker Band",
    ],
    "clothing": [
        "Cotton T-Shirt White L", "Cotton T-Shirt Black M", "Cotton T-Shirt Navy S",
        "V-Neck Tee Grey XL", "Polo Shirt Blue M", "Henley Shirt Green L",
        "Crew Neck Sweater Navy", "Zip Hoodie Grey M", "Pullover Hoodie Black L",
        "Fleece Jacket Blue XL", "Denim Jacket Classic M", "Rain Jacket Packable L",
        "Chino Pants Khaki 32", "Jogger Pants Black M", "Cargo Shorts Olive L",
        "Athletic Shorts Navy S", "Swim Trunks Blue M", "Pajama Pants Plaid L",
        "Crew Socks White 6pk", "Ankle Socks Black 6pk", "Athletic Socks 3pk",
        "Boxer Briefs 4pk M", "Brief Underwear 3pk L", "Undershirt V-Neck 3pk",
        "Beanie Hat Black", "Baseball Cap Navy", "Bucket Hat Khaki",
        "Winter Gloves Touchscreen", "Scarf Merino Wool Grey", "Belt Leather Brown 34",
        "Dress Shirt White 16", "Tie Silk Navy", "Bow Tie Black",
        "Sneakers White Size 10", "Running Shoes Black 9", "Sandals Brown Size 11",
    ],
    "general": [
        "Notebook Spiral A4", "Ballpoint Pens 10pk", "Highlighters Neon 5pk", "Sticky Notes 3x3 12pk",
        "Scissors Stainless 8in", "Tape Dispenser Clear", "Glue Stick 4pk", "Stapler Desktop",
        "Paper Clips 200ct", "Binder Clips Assorted", "Manila Folders 25pk", "Hanging Files 10pk",
        "Wall Calendar 2025", "Desk Organizer Wood", "Pencil Case Canvas", "Ruler 30cm Plastic",
        "Yoga Mat 6mm", "Resistance Bands 5pk", "Jump Rope Adjustable", "Foam Roller 18in",
        "Water Bottle 750ml", "Travel Mug Insulated", "Lunch Box Bento", "Ice Pack Reusable 2pk",
        "Umbrella Compact Black", "Tote Bag Canvas", "Backpack Daypack 20L", "Duffel Bag 40L",
        "Picture Frame 8x10", "Photo Album 200-Slot", "Gift Wrapping Paper 3pk", "Greeting Cards 10pk",
        "Candle Soy Lavender", "Incense Sticks 50pk", "Essential Oil Lavender", "Diffuser Ceramic",
        "Pet Bowl Stainless M", "Dog Leash 6ft", "Cat Toy Mouse 3pk", "Pet Bed Cushion Small",
        "Garden Gloves Pair", "Plant Pot Ceramic 6in", "Potting Soil 5L", "Watering Can 2L",
        "First Aid Kit 100pc", "Bandages Assorted 30ct", "Hand Sanitizer 500ml", "Face Masks 50pk",
    ],
    "heavy": [
        "Claw Hammer 16oz", "Ball Peen Hammer 12oz", "Rubber Mallet 16oz",
        "Adjustable Wrench 10in", "Socket Set 40pc", "Torque Wrench 1/2in",
        "Cordless Drill 20V", "Impact Driver Set", "Circular Saw 7-1/4in",
        "Jigsaw Variable Speed", "Orbital Sander", "Angle Grinder 4-1/2in",
        "Screwdriver Set 20pc", "Hex Key Set Metric", "Pliers Set 3pc",
        "Wire Stripper", "Utility Knife Heavy", "Tin Snips Aviation",
        "Tape Measure 25ft", "Speed Square 7in", "Level 24in Magnetic",
        "Stud Finder Digital", "Laser Level Cross-Line", "Chalk Line 100ft",
        "Work Gloves Leather XL", "Safety Glasses Clear", "Ear Protection Muffs",
        "Tool Box Steel 20in", "Tool Belt 11-Pocket", "Knee Pads Professional",
        "Bolt Cutters 24in", "Pry Bar Flat 12in", "C-Clamp Set 6pc",
        "Pipe Wrench 14in", "Chain Hoist 1-Ton", "Floor Jack 3-Ton",
    ],
    "beverage": [
        "Spring Water 500ml", "Sparkling Water Lemon 1L", "Mineral Water 1.5L",
        "Cola Classic 330ml", "Diet Cola 330ml", "Ginger Ale 330ml",
        "Orange Soda 330ml", "Lemon-Lime Soda 330ml", "Root Beer 330ml",
        "Orange Juice Fresh 1L", "Apple Juice 1L", "Cranberry Juice 1L",
        "Mango Nectar 1L", "Pineapple Juice 1L", "Grape Juice 500ml",
        "Green Tea Matcha Bags 20ct", "Black Tea English 40ct", "Chamomile Tea 20ct",
        "Earl Grey Tea 25ct", "Peppermint Tea 20ct", "Herbal Tea Sampler",
        "Instant Coffee 200g", "Ground Coffee Medium 250g", "Coffee Beans Dark 500g",
        "Espresso Pods 10ct", "Cold Brew Concentrate 1L", "Hot Chocolate Mix 500g",
        "Energy Drink Original 250ml", "Energy Drink Sugar-Free 250ml",
        "Sports Drink Lemon 500ml", "Protein Shake Chocolate 330ml",
        "Coconut Water 500ml", "Aloe Vera Drink 500ml", "Kombucha Ginger 330ml",
        "Lemonade Fresh 1L", "Iced Tea Peach 500ml",
    ],
}


def _generate_product_name(category: str) -> str:
    """Return a realistic product name for the given category."""
    names = PRODUCT_NAMES.get(category, PRODUCT_NAMES["general"])
    return random.choice(names)


ITEM_SIZE_PROFILES = {
    "grocery": {"weight": (0.1, 3.0), "dims": (5, 30)},
    "beverage": {"weight": (0.3, 5.0), "dims": (8, 30)},
    "perishable": {"weight": (0.1, 2.0), "dims": (5, 20)},
    "refrigerated": {"weight": (0.1, 2.0), "dims": (5, 20)},
    "chemical": {"weight": (0.2, 5.0), "dims": (5, 30)},
    "household": {"weight": (0.1, 3.0), "dims": (5, 35)},
    "flammable": {"weight": (0.1, 2.0), "dims": (5, 20)},
    "hazardous": {"weight": (0.1, 2.0), "dims": (5, 20)},
    "electronics": {"weight": (0.05, 2.0), "dims": (3, 25)},
    "fragile": {"weight": (0.05, 1.0), "dims": (3, 20)},
    "clothing": {"weight": (0.1, 1.5), "dims": (10, 40)},
    "heavy": {"weight": (1.0, 15.0), "dims": (10, 50)},
    "industrial": {"weight": (0.5, 10.0), "dims": (8, 40)},
    "general": {"weight": (0.1, 5.0), "dims": (5, 35)},
}


def _tag_to_zone_indices(tag: str) -> list[int]:
    """Return zone indices whose tags contain the given tag."""
    return [
        i for i, z in enumerate(ZONE_DEFINITIONS)
        if tag in z["tags"]
    ]


def _generate_item_dimensions(tags: list[str]) -> dict:
    """Generate realistic item dimensions based on tags."""
    primary_tag = tags[0] if tags else "general"
    profile = ITEM_SIZE_PROFILES.get(primary_tag, ITEM_SIZE_PROFILES["general"])
    weight = round(random.uniform(*profile["weight"]), 2)
    w = round(random.uniform(*profile["dims"]), 1)
    h = round(random.uniform(*profile["dims"]), 1)
    d = round(random.uniform(*profile["dims"]), 1)
    return {"weight_kg": weight, "width_cm": w, "height_cm": h, "depth_cm": d}


def generate_zones(session: Session) -> list[Zone]:
    """Create 10 warehouse zones."""
    zones = []
    for zdef in ZONE_DEFINITIONS:
        zone = Zone(
            name=zdef["name"],
            tags_json=json.dumps(zdef["tags"]),
            distance_to_picking_area=zdef["distance"],
        )
        zones.append(zone)
    session.add_all(zones)
    session.flush()
    return zones


def generate_racks(session: Session, zones: list[Zone]) -> list[Rack]:
    """Create ~500 racks distributed across zones, proportionally."""
    rack_distribution = {
        "Grocery-A": 80, "Grocery-B": 60, "Dairy-Refrigerated": 50,
        "Beverages": 50, "Household-Chemicals": 40,
        "Flammable-Storage": 30, "Electronics": 50,
        "Clothing-Apparel": 50, "Heavy-Industrial": 30,
        "General-Merchandise": 60,
    }
    racks = []
    for zone in zones:
        n_racks = rack_distribution.get(zone.name, 40)
        for i in range(n_racks):
            num_shelves = random.randint(4, 10)
            is_heavy = "heavy" in zone.tags
            rack = Rack(
                zone_id=zone.id,
                name=f"{zone.name}-R{i+1:03d}",
                max_weight_kg=500.0 if is_heavy else random.uniform(100.0, 300.0),
                num_shelves=num_shelves,
                shelf_height_cm=random.uniform(30.0, 60.0),
                shelf_width_cm=random.uniform(80.0, 150.0),
                shelf_depth_cm=random.uniform(40.0, 80.0),
            )
            racks.append(rack)
    session.add_all(racks)
    session.flush()
    return racks


def generate_slots(session: Session, racks: list[Rack]) -> list[int]:
    """Create slots for each rack. Returns list of slot IDs.

    Slots get (x, y) coordinates based on rack position in the warehouse grid.
    Racks are laid out in a grid pattern within each zone.
    """
    slot_ids = []
    batch = []
    batch_size = 5000

    rack_counter_by_zone: dict[int, int] = {}
    for rack in _safe_tqdm(racks, desc="Generating slots"):
        zone_idx = rack_counter_by_zone.get(rack.zone_id, 0)
        rack_counter_by_zone[rack.zone_id] = zone_idx + 1

        grid_cols = 10
        row = zone_idx // grid_cols
        col = zone_idx % grid_cols

        base_x = (rack.zone_id - 1) * 100 + col * 10
        base_y = row * 5

        positions_per_shelf = max(1, int(rack.shelf_width_cm / 25))
        weight_per_slot = round(rack.max_weight_kg / rack.num_shelves / positions_per_shelf, 2)

        for shelf_num in range(1, rack.num_shelves + 1):
            for pos in range(1, positions_per_shelf + 1):
                slot = Slot(
                    rack_id=rack.id,
                    shelf_number=shelf_num,
                    position_on_shelf=pos,
                    width_cm=round(rack.shelf_width_cm / positions_per_shelf, 1),
                    height_cm=round(rack.shelf_height_cm, 1),
                    depth_cm=round(rack.shelf_depth_cm, 1),
                    current_weight_kg=0.0,
                    max_weight_kg=weight_per_slot,
                    x_coord=base_x + pos * 2.0,
                    y_coord=base_y + shelf_num * 1.5,
                )
                batch.append(slot)

        if len(batch) >= batch_size:
            session.add_all(batch)
            session.flush()
            slot_ids.extend(s.id for s in batch)
            batch = []

    if batch:
        session.add_all(batch)
        session.flush()
        slot_ids.extend(s.id for s in batch)

    return slot_ids


def generate_items(session: Session, n_items: int = 100_000) -> list[Item]:
    """Create items, mixing affinity-group items with random filler items."""
    items = []
    sku_counter = 0

    affinity_group_items = []
    for group_name, group in AFFINITY_GROUPS.items():
        for item_name in group["items"]:
            sku_counter += 1
            dims = _generate_item_dimensions(group["tags"])
            item = Item(
                sku=f"SKU-{sku_counter:06d}",
                name=item_name,
                tags_json=json.dumps(group["tags"]),
                **dims,
            )
            affinity_group_items.append(item)

    items.extend(affinity_group_items)

    all_tag_sets = list(ITEM_SIZE_PROFILES.keys())
    remaining = n_items - len(items)

    filler_categories = [
        ("grocery", ["grocery"]),
        ("household", ["chemical", "household"]),
        ("electronics", ["electronics", "fragile"]),
        ("clothing", ["clothing"]),
        ("general", ["general"]),
        ("heavy", ["heavy", "industrial"]),
        ("beverage", ["grocery", "beverage"]),
    ]

    for i in _safe_tqdm(range(remaining), desc="Generating items"):
        sku_counter += 1
        cat_name, tags = random.choice(filler_categories)
        dims = _generate_item_dimensions(tags)
        item = Item(
            sku=f"SKU-{sku_counter:06d}",
            name=_generate_product_name(cat_name),
            tags_json=json.dumps(tags),
            **dims,
        )
        items.append(item)

    random.shuffle(items)

    batch_size = 10_000
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        session.add_all(batch)
        session.flush()

    return items


def generate_order_history(
    session: Session,
    items: list[Item],
    n_orders: int = 100_000,
    target_records: int = 500_000,
) -> None:
    """Generate order history with realistic co-purchase patterns.

    Co-purchase patterns:
    - Items from the same affinity group appear together frequently
    - Some random cross-group purchases
    - Velocity follows a power-law: a few items are ordered very frequently
    """
    item_id_by_name: dict[str, int] = {}
    affinity_item_ids: dict[str, list[int]] = {}

    for group_name, group in AFFINITY_GROUPS.items():
        affinity_item_ids[group_name] = []
        for item_name in group["items"]:
            for item in items:
                if item.name == item_name:
                    item_id_by_name[item_name] = item.id
                    affinity_item_ids[group_name].append(item.id)
                    break

    all_item_ids = [item.id for item in items]

    # Power-law distribution for item popularity:
    # top 20% items appear in 60% of orders
    n_items = len(all_item_ids)
    popularity = np.random.zipf(1.5, n_items)
    popularity = popularity / popularity.sum()

    hot_items = np.random.choice(all_item_ids, size=int(n_items * 0.2), replace=False)

    records_generated = 0
    batch = []
    batch_size = 10_000
    base_date = datetime(2025, 1, 1)

    group_names = list(affinity_item_ids.keys())

    pbar = _safe_tqdm(total=target_records, desc="Generating orders")
    order_num = 0

    while records_generated < target_records:
        order_num += 1
        order_id = f"ORD-{order_num:07d}"
        order_date = base_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(8, 22),
            minutes=random.randint(0, 59),
        )

        items_in_order = random.randint(2, 8)
        order_item_ids = set()

        # 70% chance: pick items from same affinity group (co-purchase pattern)
        if random.random() < 0.7 and group_names:
            group = random.choice(group_names)
            group_items = affinity_item_ids.get(group, [])
            if group_items:
                n_from_group = min(len(group_items), random.randint(2, 5))
                order_item_ids.update(random.sample(group_items, n_from_group))

        # Fill remaining with popularity-weighted random items
        while len(order_item_ids) < items_in_order:
            if random.random() < 0.6 and len(hot_items) > 0:
                order_item_ids.add(int(np.random.choice(hot_items)))
            else:
                order_item_ids.add(random.choice(all_item_ids))

        for item_id in order_item_ids:
            record = OrderHistory(
                order_id=order_id,
                item_id=item_id,
                quantity=random.randint(1, 5),
                ordered_at=order_date,
            )
            batch.append(record)
            records_generated += 1
            pbar.update(1)

        if len(batch) >= batch_size:
            session.add_all(batch)
            session.flush()
            batch = []

        if records_generated >= target_records:
            break

    if batch:
        session.add_all(batch)
        session.flush()

    pbar.close()


def generate_all_test_data(session: Session) -> dict:
    """Generate the complete test dataset and return summary stats."""
    print("\n=== Generating Warehouse Test Data ===\n")

    print("[1/4] Creating zones...")
    zones = generate_zones(session)
    print(f"  Created {len(zones)} zones")

    print("[2/4] Creating racks and slots...")
    racks = generate_racks(session, zones)
    print(f"  Created {len(racks)} racks")
    slot_ids = generate_slots(session, racks)
    print(f"  Created {len(slot_ids)} slots")

    print("[3/4] Creating items...")
    items = generate_items(session, n_items=100_000)
    print(f"  Created {len(items)} items")

    print("[4/4] Creating order history...")
    generate_order_history(session, items, n_orders=100_000, target_records=500_000)

    session.commit()

    return {
        "zones": len(zones),
        "racks": len(racks),
        "slots": len(slot_ids),
        "items": len(items),
    }

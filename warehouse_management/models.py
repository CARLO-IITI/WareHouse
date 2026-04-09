"""SQLAlchemy ORM models for the warehouse management system."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Zone(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    tags_json = Column(Text, nullable=False, default="[]")
    distance_to_picking_area = Column(Float, nullable=False, default=0.0)

    racks = relationship("Rack", back_populates="zone", lazy="selectin")

    @property
    def tags(self) -> list[str]:
        return json.loads(self.tags_json)

    @tags.setter
    def tags(self, value: list[str]):
        self.tags_json = json.dumps(value)

    def __repr__(self):
        return f"<Zone(id={self.id}, name='{self.name}', tags={self.tags})>"


class Rack(Base):
    __tablename__ = "racks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    max_weight_kg = Column(Float, nullable=False)
    num_shelves = Column(Integer, nullable=False)
    shelf_height_cm = Column(Float, nullable=False)
    shelf_width_cm = Column(Float, nullable=False)
    shelf_depth_cm = Column(Float, nullable=False)

    zone = relationship("Zone", back_populates="racks")
    slots = relationship("Slot", back_populates="rack", lazy="selectin")

    def __repr__(self):
        return f"<Rack(id={self.id}, name='{self.name}', zone_id={self.zone_id})>"


class Slot(Base):
    __tablename__ = "slots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rack_id = Column(Integer, ForeignKey("racks.id"), nullable=False)
    shelf_number = Column(Integer, nullable=False)
    position_on_shelf = Column(Integer, nullable=False)
    width_cm = Column(Float, nullable=False)
    height_cm = Column(Float, nullable=False)
    depth_cm = Column(Float, nullable=False)
    current_weight_kg = Column(Float, nullable=False, default=0.0)
    max_weight_kg = Column(Float, nullable=False)
    x_coord = Column(Float, nullable=False)
    y_coord = Column(Float, nullable=False)

    rack = relationship("Rack", back_populates="slots")
    item = relationship("Item", back_populates="slot", uselist=False)

    __table_args__ = (
        Index("ix_slots_rack_id", "rack_id"),
        Index("ix_slots_coords", "x_coord", "y_coord"),
    )


    @property
    def remaining_weight_capacity(self) -> float:
        return self.max_weight_kg - self.current_weight_kg

    @property
    def is_occupied(self) -> bool:
        return self.item is not None

    def __repr__(self):
        return (
            f"<Slot(id={self.id}, rack={self.rack_id}, "
            f"shelf={self.shelf_number}, pos={self.position_on_shelf})>"
        )


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku = Column(String(50), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    tags_json = Column(Text, nullable=False, default="[]")
    weight_kg = Column(Float, nullable=False)
    width_cm = Column(Float, nullable=False)
    height_cm = Column(Float, nullable=False)
    depth_cm = Column(Float, nullable=False)
    velocity_score = Column(Float, nullable=False, default=0.0)
    velocity_class = Column(String(1), nullable=False, default="C")
    current_slot_id = Column(
        Integer, ForeignKey("slots.id"), nullable=True,
    )

    slot = relationship("Slot", back_populates="item", foreign_keys=[current_slot_id])

    __table_args__ = (
        Index("ix_items_velocity", "velocity_score"),
        Index("ix_items_slot", "current_slot_id"),
    )

    @property
    def tags(self) -> list[str]:
        return json.loads(self.tags_json)

    @tags.setter
    def tags(self, value: list[str]):
        self.tags_json = json.dumps(value)

    @property
    def is_assigned(self) -> bool:
        return self.current_slot_id is not None

    def __repr__(self):
        return f"<Item(id={self.id}, sku='{self.sku}', name='{self.name}')>"


class OrderHistory(Base):
    __tablename__ = "order_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), nullable=False)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    ordered_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_order_history_order_id", "order_id"),
        Index("ix_order_history_item_id", "item_id"),
        Index("ix_order_history_ordered_at", "ordered_at"),
    )

    def __repr__(self):
        return (
            f"<OrderHistory(id={self.id}, order_id='{self.order_id}', "
            f"item_id={self.item_id})>"
        )

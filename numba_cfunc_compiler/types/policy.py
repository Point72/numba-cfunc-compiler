"""Typed policies for host-backed payloads."""

from enum import Enum


class ValueSemantics(Enum):
    COPY = "copy"
    BORROWED_VIEW = "borrowed_view"

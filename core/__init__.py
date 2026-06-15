"""Modelos de dominio e mecanismos centrais (kill switch)."""

from core.kill_switch import (
    DEFAULT_SENTINEL,
    KillSwitch,
    KillSwitchEngagedError,
)
from core.models import (
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    Signal,
    TimeInForce,
)

__all__ = [
    "DEFAULT_SENTINEL",
    "KillSwitch",
    "KillSwitchEngagedError",
    "OrderIntent",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "Position",
    "Signal",
    "TimeInForce",
]

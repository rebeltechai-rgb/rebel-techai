from __future__ import annotations

import importlib
from typing import Any

from .base import BrokerAdapter
from .mt5_adapter import MT5Adapter


def _load_custom_adapter(spec: str) -> BrokerAdapter:
    """
    Load adapter from "module:ClassName" or "module" (defaults to Adapter).
    """
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
    else:
        module_name, class_name = spec, "Adapter"
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls()


def get_adapter(config: dict) -> BrokerAdapter:
    adapter_cfg = config.get("broker_adapter", {})
    name = (adapter_cfg.get("name") or "mt5").lower()
    module_spec = adapter_cfg.get("module")

    if module_spec:
        return _load_custom_adapter(module_spec)
    if name == "mt5":
        return MT5Adapter()

    raise ValueError(f"Unknown broker adapter: {name}")

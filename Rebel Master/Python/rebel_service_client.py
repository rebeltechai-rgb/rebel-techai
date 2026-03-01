"""
REBEL Service Client
====================
Small helper used by GUIs / tools to talk to rebel_service.py
via service_control.json.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

BASE_DIR = r"C:\Rebel Technologies\Rebel Master"
CONFIG_DIR = os.path.join(BASE_DIR, "Config")
SERVICE_CONTROL_PATH = os.path.join(CONFIG_DIR, "service_control.json")


def _load_control() -> Dict[str, Any]:
    if not os.path.exists(SERVICE_CONTROL_PATH):
        return {"command": None, "updated_at": None}
    try:
        with open(SERVICE_CONTROL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # if corrupted, just reset
        return {"command": None, "updated_at": None}


def _save_control(data: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp_path = SERVICE_CONTROL_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, SERVICE_CONTROL_PATH)


def send_command(command: str) -> None:
    """
    command: "start", "stop", or "restart"
    """
    command = command.lower()
    if command not in ("start", "stop", "restart"):
        raise ValueError(f"Invalid command: {command}")

    data = _load_control()
    data["command"] = command
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_control(data)
    print(f"[SERVICE_CLIENT] Sent command: {command}")


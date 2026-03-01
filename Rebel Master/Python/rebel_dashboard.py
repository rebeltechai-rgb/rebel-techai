from flask import Flask, request, jsonify, send_from_directory
import yaml
import os
import json
from datetime import datetime

CONFIG_PATH = r"C:\Rebel Technologies\Rebel Master\Config\master_config.yaml"
SYMBOL_LIST_PATH = r"C:\Rebel Technologies\Rebel Master\Config\symbol_lists.yaml"
STATE_PATH = r"C:\Rebel Technologies\Rebel Master\State\runtime_state.json"

app = Flask(__name__)


# -------------------------
# Helpers
# -------------------------

def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH, "r") as f:
        return json.load(f)


# -------------------------
# ROUTES
# -------------------------

@app.route("/")
def root():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    config = load_yaml(CONFIG_PATH)
    return jsonify(config)


@app.route("/api/config/update", methods=["POST"])
def update_config():
    data = request.json
    config = load_yaml(CONFIG_PATH)

    # merge updates into config
    for key, value in data.items():
        config[key] = value

    save_yaml(CONFIG_PATH, config)
    return jsonify({"ok": True, "updated": data})


@app.route("/api/symbols", methods=["GET"])
def get_symbols():
    symbols = load_yaml(SYMBOL_LIST_PATH)
    return jsonify(symbols)


@app.route("/api/symbols/add", methods=["POST"])
def add_symbol():
    data = request.json
    symbol = data.get("symbol")

    symbols = load_yaml(SYMBOL_LIST_PATH)
    groups = symbols.get("groups", {})

    if "custom" not in groups:
        groups["custom"] = []

    if symbol not in groups["custom"]:
        groups["custom"].append(symbol)

    symbols["groups"] = groups
    save_yaml(SYMBOL_LIST_PATH, symbols)

    return jsonify({"ok": True, "added": symbol})


@app.route("/api/symbols/remove", methods=["POST"])
def remove_symbol():
    data = request.json
    symbol = data.get("symbol")

    symbols = load_yaml(SYMBOL_LIST_PATH)
    groups = symbols.get("groups", {})

    for group in groups:
        if symbol in groups[group]:
            groups[group].remove(symbol)

    symbols["groups"] = groups
    save_yaml(SYMBOL_LIST_PATH, symbols)

    return jsonify({"ok": True, "removed": symbol})


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(load_state())


# -------------------------
# Run server
# -------------------------

if __name__ == "__main__":
    print("REBEL Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

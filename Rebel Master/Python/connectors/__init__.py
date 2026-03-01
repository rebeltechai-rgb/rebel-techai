from .base_connector import BrokerConnector
from .mt5_connector import MT5Connector


def create_broker(config: dict) -> BrokerConnector:
    """
    Factory function that returns the correct broker connector
    based on config["broker"]["type"].
    """
    broker_type = config.get("broker", {}).get("type", "mt5").lower()

    if broker_type == "mt5":
        return MT5Connector()

    # future connectors:
    # if broker_type == "oanda": return OandaConnector()
    # if broker_type == "ibkr": return InteractiveBrokersConnector()

    raise ValueError(f"Unsupported broker type: {broker_type}")

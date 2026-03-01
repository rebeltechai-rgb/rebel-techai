"""
REBEL Notifier - Notification Module
Handles Telegram alerts for trade signals and execution.
"""

import requests
from typing import Optional


class RebelNotifier:
    """Notification handler for Telegram alerts."""
    
    def __init__(self, config: dict):
        """
        Initialize the notifier with configuration.
        
        Args:
            config: Master configuration dictionary
        """
        telegram_cfg = config.get("telegram", {})
        
        self.enabled = telegram_cfg.get("enabled", False)
        self.bot_token = telegram_cfg.get("bot_token", "")
        self.chat_id = telegram_cfg.get("chat_id", "")
    
    def send_telegram(self, text: str) -> bool:
        """
        Send a message via Telegram bot.
        
        Args:
            text: Message text to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        # Check if Telegram is enabled and configured
        if not self.enabled:
            return False
        
        if not self.bot_token or not self.chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception:
            # Silently fail - do not crash the bot
            return False
    
    def notify_signal(self, symbol: str, direction: str, score: int, balance: float) -> bool:
        """
        Send a trade signal notification.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            score: Signal score
            balance: Current account balance
            
        Returns:
            True if sent successfully
        """
        text = (
            f"🤖 <b>REBEL TRADE</b>\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Score: {score}/5\n"
            f"Balance: ${balance:,.2f}"
        )
        return self.send_telegram(text)
    
    def notify_trade_opened(self, symbol: str, direction: str, volume: float, 
                            price: float, sl: float, tp: float, ticket: int) -> bool:
        """
        Send a trade execution notification.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction
            volume: Trade volume
            price: Entry price
            sl: Stop loss price
            tp: Take profit price
            ticket: Order ticket number
            
        Returns:
            True if sent successfully
        """
        text = (
            f"✅ <b>TRADE OPENED</b>\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Volume: {volume}\n"
            f"Price: {price}\n"
            f"SL: {sl}\n"
            f"TP: {tp}\n"
            f"Ticket: #{ticket}"
        )
        return self.send_telegram(text)
    
    def notify_error(self, message: str) -> bool:
        """
        Send an error notification.
        
        Args:
            message: Error message
            
        Returns:
            True if sent successfully
        """
        text = f"⚠️ <b>REBEL ERROR</b>\n{message}"
        return self.send_telegram(text)
    
    def notify_startup(self, account_login: int, balance: float) -> bool:
        """
        Send a bot startup notification.
        
        Args:
            account_login: MT5 account login
            balance: Account balance
            
        Returns:
            True if sent successfully
        """
        text = (
            f"🚀 <b>REBEL BOT STARTED</b>\n"
            f"Account: {account_login}\n"
            f"Balance: ${balance:,.2f}"
        )
        return self.send_telegram(text)


# Allow running standalone for testing
if __name__ == "__main__":
    print("RebelNotifier - Test Mode")
    
    # Example config (tokens redacted)
    test_config = {
        "telegram": {
            "enabled": False,
            "bot_token": "YOUR_BOT_TOKEN",
            "chat_id": "YOUR_CHAT_ID"
        }
    }
    
    notifier = RebelNotifier(test_config)
    print(f"Telegram enabled: {notifier.enabled}")
    print(f"Bot token set: {bool(notifier.bot_token)}")
    print(f"Chat ID set: {bool(notifier.chat_id)}")


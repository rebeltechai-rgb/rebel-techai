#!/usr/bin/env python3
"""
REBEL RULES-BASED AI TRADING SYSTEM
Entry Point
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rebel_core.engine import run_engine


def main():
    """Main entry point."""
    # Default config path
    config_path = "config/rebel_config.yaml"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Check for OpenAI key
    if not os.getenv('OPENAI_API_KEY'):
        print("[WARNING] OPENAI_API_KEY not set")
        print("  AI Brain will be disabled, using score-only mode")
        print("  To enable AI, set: export OPENAI_API_KEY=your-key")
        print()
    
    # Run engine
    run_engine(config_path)


if __name__ == "__main__":
    main()



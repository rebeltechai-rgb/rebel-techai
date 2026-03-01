#!/usr/bin/env python
"""Apply the ML bypass patch to rebel_engine.py"""

# Read the current file
with open('rebel_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if 'ml_bypass_symbols' in content:
    print('Already patched!')
    exit(0)

# Find the location to insert the bypass code
old_text = '        self.ml_shadow_mode = not self.ml_enabled  # shadow_mode = True means don\'t block\n        \n        # Select model based on config'

new_text = '''        self.ml_shadow_mode = not self.ml_enabled  # shadow_mode = True means don't block
        
        # Build ML bypass symbol list from config groups
        self.ml_bypass_symbols = set()
        bypass_groups = self.ml_filter_config.get("bypass_groups", [])
        symbols_config = self.config.get("symbols", {})
        groups_config = symbols_config.get("groups", {}) if isinstance(symbols_config, dict) else {}
        
        for group_name in bypass_groups:
            group = groups_config.get(group_name, {})
            if group.get("enabled", True):
                group_symbols = group.get("symbols", [])
                self.ml_bypass_symbols.update(group_symbols)
        
        if self.ml_bypass_symbols:
            print(f"[ML] Bypass enabled for {len(self.ml_bypass_symbols)} symbols from groups: {bypass_groups}")
        
        # Select model based on config'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('rebel_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: Bypass code added to __init__!')
else:
    print('ERROR: Could not find exact insertion point')
    # Debug info
    if "self.ml_shadow_mode" in content:
        idx = content.find("self.ml_shadow_mode")
        print(f"Found at index {idx}")
        print("Context around it:")
        print(repr(content[idx:idx+200]))

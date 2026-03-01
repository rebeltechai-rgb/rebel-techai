#!/usr/bin/env python
"""Apply the bypass_groups to master_config.yaml"""
import os

config_path = r'C:\Rebel Technologies\Rebel Master\Config\master_config.yaml'

# Read the current file
with open(config_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if 'bypass_groups' in content:
    print('Already patched!')
    exit(0)

# Find and replace
old_text = '''ml_filter:
  enabled: true              # MASTER SWITCH: true = block bad trades, false = shadow mode only
  model: "rf_v2"             # model to use: rf_v1 or rf_v2
  min_win_prob: 0.50         # minimum P(win) to allow trade (0.50 = 50%)
  max_loss_prob: 0.70        # reject if P(loss) above this
  log_decisions: true        # log all ML decisions to file
  
  # Hard filters (applied on top of ML probability)
  hard_filters:'''

new_text = '''ml_filter:
  enabled: true              # MASTER SWITCH: true = block bad trades, false = shadow mode only
  model: "rf_v2"             # model to use: rf_v1 or rf_v2
  min_win_prob: 0.50         # minimum P(win) to allow trade (0.50 = 50%)
  max_loss_prob: 0.70        # reject if P(loss) above this
  log_decisions: true        # log all ML decisions to file
  
  # Asset classes to BYPASS ML filter (these trade without ML gating)
  bypass_groups:
    - commodities            # METALS: 100% WR - let them trade freely
    - indices                # INDICES: 85% WR - let them trade freely
  
  # Hard filters (applied on top of ML probability)
  hard_filters:'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: bypass_groups added to config!')
else:
    print('ERROR: Could not find insertion point')
    if 'ml_filter:' in content:
        idx = content.find('ml_filter:')
        print("ml_filter section:")
        print(content[idx:idx+500])

#!/usr/bin/env python
"""Apply the ML bypass check to the run loop in rebel_engine.py"""

# Read the current file
with open('rebel_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if 'ML:BYPASS' in content:
    print('Already patched!')
    exit(0)

# Find and replace the ML evaluation section
old_text = '''                            # ============================================
                            # ML SHADOW MODE EVALUATION
                            # ============================================
                            ml_result = None
                            if self.ml_filter is not None and "features" in signal:'''

new_text = '''                            # ============================================
                            # ML FILTER EVALUATION (with bypass for winning asset classes)
                            # ============================================
                            ml_result = None
                            
                            # Check if symbol is in bypass list (METALS, INDICES, etc.)
                            if symbol in self.ml_bypass_symbols:
                                print(f"[ML:BYPASS] {symbol} - bypassing ML filter (winning asset class)")
                            elif self.ml_filter is not None and "features" in signal:'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('rebel_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: Bypass check added to run loop!')
else:
    print('ERROR: Could not find exact insertion point')
    if "ML SHADOW MODE" in content:
        print("Found 'ML SHADOW MODE' - checking format...")
    else:
        print("'ML SHADOW MODE' not found")

#!/usr/bin/env python
"""Test asset class detection and score thresholds."""
import sys
sys.path.insert(0, r'C:\Rebel Technologies\Rebel Master\ML')

from rebel_signals import RebelSignals

class MockBroker:
    pass

signals = RebelSignals(MockBroker())

test_symbols = [
    ('EURUSD', 'fx_major'),
    ('GBPJPY', 'fx_minor'),
    ('XAUUSD', 'metals'),
    ('BRENT.fs', 'energies'),
    ('UKOIL', 'energies'),
    ('COFFEE.fs', 'softs'),
    ('COCOA.fs', 'softs'),
    ('US500', 'indices'),
    ('GER40', 'indices'),
    ('USTECH', 'indices'),
    ('BTCUSD', 'crypto'),
]

print('=== ASSET CLASS DETECTION ===')
all_pass = True
for sym, expected in test_symbols:
    actual = signals._get_asset_class(sym)
    score = signals._get_score_threshold(sym)
    status = 'OK' if actual == expected else 'FAIL'
    if actual != expected:
        all_pass = False
    print(f'{status} {sym:15} -> {actual:12} (min_score={score})')

print()
print('=== SCORE THRESHOLDS SUMMARY ===')
print('Metals/Energies/Softs: 3 (more trades for 100% WR assets)')
print('Indices:               3 (more trades for 85% WR assets)')
print('Crypto:                4 (moderate for 49% WR)')
print('FX:                    4 (stricter for 30% WR)')
print()
if all_pass:
    print('SUCCESS: All asset classes detected correctly!')
else:
    print('ERROR: Some asset classes not detected correctly')

import csv

# Check for COFFEE in trade_features
print('COFFEE in trade_features.csv:')
print('-'*60)
coffee_features = []
with open(r'C:\Rebel Technologies\Rebel Master\ML\trade_features.csv', 'r') as f:
    for row in csv.DictReader(f):
        if 'COFFEE' in row['symbol'].upper():
            coffee_features.append(row['ticket'])
            print(f"  {row['timestamp']} | {row['symbol']} | Ticket: {row['ticket']}")

if not coffee_features:
    print("  (none found)")

print()
print('COFFEE in labels.csv:')
print('-'*60)
coffee_labels = []
with open(r'C:\Rebel Technologies\Rebel Master\ML\labels.csv', 'r') as f:
    for row in csv.DictReader(f):
        if 'COFFEE' in row['symbol'].upper():
            coffee_labels.append(row['ticket'])
            print(f"  {row['timestamp']} | {row['symbol']} | Ticket: {row['ticket']} | {row['outcome_class']}")

if not coffee_labels:
    print("  (none found)")

print()
print('='*60)
print(f"Coffee features: {len(coffee_features)}")
print(f"Coffee labels: {len(coffee_labels)}")

# Check for unmatched
feature_set = set(coffee_features)
label_set = set(coffee_labels)
unmatched_features = feature_set - label_set
unmatched_labels = label_set - feature_set

if unmatched_features:
    print(f"Features without labels: {unmatched_features}")
if unmatched_labels:
    print(f"Labels without features: {unmatched_labels}")

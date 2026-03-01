"""Quick analysis of backup dataset for splitting into specialized models"""
import pandas as pd

df = pd.read_csv('training_dataset_backup_812.csv')
print("=" * 50)
print("BACKUP DATASET ANALYSIS")
print("=" * 50)
print(f"Total trades: {len(df)}")
print()

# Crypto keywords
crypto_keys = ['BTC','ETH','XRP','ADA','SOL','DOGE','LTC','LINK','UNI','AAVE',
               'DOT','BNB','XLM','AVAX','MATIC','SHIB','ATOM','ALGO','FTM','SAND',
               'MANA','AXS','CRV','COMP','SUSHI','YFI','SNX','MKR','ENJ','BAT',
               'ZEC','DASH','XMR','EOS','TRX','XTZ','LRC','BCH','NEAR','APE']

# Metals
metals = ['XAUUSD','XAGUSD','XAUEUR','XAUGBP','XAUAUD','XPTUSD']

# Indices
indices_keys = ['US30','US500','US2000','USTECH','GER40','UK100','DAX','NAS100',
                'FT100','VIX','DJ30','S&P','AUS200','CN50','EU50','FRA40','HK50',
                'JPN225','NK225','CAC40','SPI200','EUSTX','HSI','IT40','SWI20',
                'SPA35','NETH25','SGFREE','CHINA50','USDINDEX']

# Energies
energies = ['BRENT','WTI','USOIL','UKOIL','NATGAS']

# Softs
softs = ['COCOA','COFFEE','COPPER','SOYBEAN']

def classify(symbol):
    s = symbol.upper()
    if any(k in s for k in crypto_keys) or '-USD' in s:
        return 'crypto'
    if any(m in s for m in metals):
        return 'metals'
    if any(i in s for i in indices_keys):
        return 'indices'
    if any(e in s for e in energies):
        return 'energies'
    if any(c in s for c in softs):
        return 'softs'
    return 'forex'

df['asset_class'] = df['symbol'].apply(classify)

# Breakdown
print("BREAKDOWN BY ASSET CLASS:")
print("-" * 30)
for ac in ['forex','crypto','metals','indices','energies','softs']:
    subset = df[df['asset_class'] == ac]
    wins = subset[subset['label'] == 1]
    print(f"{ac.upper():12} {len(subset):4} trades | WR: {len(wins)/len(subset)*100:.1f}%" if len(subset) > 0 else f"{ac.upper():12}    0 trades")

print()
print("CAN TRAIN SPECIALIZED MODELS:")
print("-" * 30)
print(f"Crypto Trader Pro:  {len(df[df['asset_class']=='crypto'])} trades (need 200+ ideal)")
print(f"FX Trader Pro:      {len(df[df['asset_class']=='forex'])} trades")
print(f"General (all):      {len(df)} trades")

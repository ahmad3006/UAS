from pathlib import Path
import pandas as pd

p = Path('archive/XPQRS/Pure_Sinusoidal.csv')
print('exists', p.exists())
with p.open() as f:
    lines = f.read().splitlines()
print('first 5 lines:')
for line in lines[:5]:
    print(line)

df = pd.read_csv(p)
print('shape', df.shape)
print('columns', list(df.columns))
print(df.head(3).to_string(index=False))

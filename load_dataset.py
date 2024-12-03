from datasets import load_dataset
import pandas as pd

ds = load_dataset("lsb/million-english-numbers")
df = pd.DataFrame(ds['train'][:10000])

print(df.head())

for i in range(100, 10000):
    if "hundred" in df.loc[i, 'text']:
        df.loc[i, 'text'] = df.loc[i, 'text'].replace("hundred", "hundred and")

df['number'] = range(10000)

df.to_csv('all_data.csv', index=False)
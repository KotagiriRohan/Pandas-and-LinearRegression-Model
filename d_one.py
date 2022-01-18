import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D1.csv", index_col=0)
numcol = []
for col in df.columns:
    if col in ['A', 'B', 'C']:
        continue
    else:
        numcol.append(col)
D1_1 = df[numcol].copy()
D1_1['A'] = df['A'].copy()
D1_1 = D1_1.dropna(axis=0).sort_values(by=['A'])

D1_2 = df[numcol].copy()
D1_2['B'] = df['B'].copy()
D1_2 = D1_2.dropna(axis=0).sort_values(by=['B'])

D1_3 = df[numcol].copy()
D1_3['C'] = df['C'].copy()
D1_3 = D1_3.dropna(axis=0).sort_values(by=['C'])

print(D1_1.shape, D1_2.shape, D1_3.shape)


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

sns.distplot(D1_1['A'], kde=False, ax=axes[0], label='A')
axes[0].set_title('D1_1 A')
sns.distplot(D1_2['B'], kde=False, ax=axes[1], label='B')
axes[1].set_title('D1_2 B')
sns.distplot(D1_3['C'], kde=False, ax=axes[2], label='C')
axes[2].set_title('D1_3 C')

plt.show()
plt.savefig("histograms.png")

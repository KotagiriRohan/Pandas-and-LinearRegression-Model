import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Plot.xlsx')
plt.figure(figsize=(16, 6))
plt.plot(df['X'], df['Y'], linestyle='-', color='cyan')
plt.xlabel('X')
plt.ylabel('Y')
plt.tick_params(left=False, bottom=False)
plt.title('Plot  X column as X axis and data in Y column as Y axis')
plt.show()

plt.savefig('plot.png')

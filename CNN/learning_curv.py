import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = input('input csv path >')
df = pd.read_csv(csv_path, index_col=0)

x = range(1, len(df.index)+1, 1)

plt.figure()
plt.title(f'{os.path.splitext(os.path.basename(csv_path))[0]}')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, df['loss'], label='training loss')
plt.plot(x, df['val_loss'], label='validation loss')
plt.xlim([x[0], x[-1]])
plt.ylim([0,2.5])
plt.xticks(range(0, 36, 5))
plt.yticks([i/10 for i in range(0, 26, 1)])
plt.legend()
plt.show()
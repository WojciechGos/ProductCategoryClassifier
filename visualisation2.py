import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    'epoch': [5, 10, 25, 50, 100],
    'avg_precision': [0.9617399536724766, 0.9739775640385656, 0.9779642850584791, 0.9780634902135219, 0.9782419695678651],
    'avg_recall': [0.9617399536724766, 0.9739775640385656, 0.9779642850584791, 0.9780634902135219, 0.9782419695678651],
    'correct_predictions': [16806.0, 16806.0, 16806.0, 16806.0, 16806.0]
}

df = pd.DataFrame(data)

plt.figure(figsize=(15, 5))

plt.subplot(131)
ax1 = sns.barplot(x='epoch', y='avg_precision', data=df)
plt.title('Avg Precision')
ax1.set_ylim(0.96, 0.98)
for i, v in enumerate(df['avg_precision']):
    ax1.text(i, v, f'{v:.5f}', ha='center', va='bottom')  

plt.subplot(132)
ax2 = sns.barplot(x='epoch', y='avg_recall', data=df)
plt.title('Avg Recall')
ax2.set_ylim(0.96, 0.98)  
for i, v in enumerate(df['avg_recall']):
    ax2.text(i, v, f'{v:.5f}', ha='center', va='bottom')  

plt.subplot(133)
ax3 = sns.barplot(x='epoch', y='correct_predictions', data=df)
plt.title('Correct Predictions')
for i, v in enumerate(df['correct_predictions']):
    ax3.text(i, v, f'{v:.1f}', ha='center', va='bottom')  

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.title('Różnice między Epoch')

max_precision = df['avg_precision'].max()
max_recall = df['avg_recall'].max()

df['precision_diff'] = max_precision - df['avg_precision']
df['recall_diff'] = max_recall - df['avg_recall']

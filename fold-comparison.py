import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")

folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Avg']
correct_predictions = [16808, 16805, 16805, 16806]
precision = [0.9756068538791052, 0.9762570663493008, 0.9760785480511752, 0.9760785480511752]
recall = [0.9756068538791052, 0.9762570663493008, 0.9760785480511752, 0.9759808227598604]

index = np.arange(len(folds))

dark_yellow = '#DAA520'

colors_correct_predictions = ['#FF6347', '#FF6347', '#FF6347', dark_yellow]  # Czerwony i ciemny żółty
colors_precision = ['#4682B4', '#4682B4', '#4682B4', dark_yellow]  # Niebieski i ciemny żółty
colors_recall = ['#32CD32', '#32CD32', '#32CD32', dark_yellow]  # Zielony i ciemny żółty

plt.figure(figsize=(10, 6))
bars = plt.bar(index, correct_predictions, color=colors_correct_predictions, edgecolor='black', linewidth=1.5)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Liczba Poprawnych Prognoz', fontsize=12)
plt.title('Poprawne Prognozy', fontsize=14)
plt.xticks(index, folds, fontsize=10)
plt.ylim(16803, 16810)
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', fontsize=11)

plt.show()

plt.figure(figsize=(10, 6))
bars = plt.bar(index, precision, color=colors_precision, edgecolor='black', linewidth=1.5)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Precyzja', fontsize=12)
plt.title('Precyzja', fontsize=14)
plt.xticks(index, folds, fontsize=10)
plt.ylim([0.97, 0.977])
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0002, f'{yval:.6f}', ha='center', fontsize=11)

plt.show()

plt.figure(figsize=(10, 6))
bars = plt.bar(index, recall, color=colors_recall, edgecolor='black', linewidth=1.5)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.title('Recall', fontsize=14)
plt.xticks(index, folds, fontsize=10)
plt.ylim([0.97, 0.977]) 
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0002, f'{yval:.6f}', ha='center', fontsize=11)

plt.show()

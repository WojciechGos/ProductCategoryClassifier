import matplotlib.pyplot as plt
import fasttext
import pandas as pd
import re
from sklearn.model_selection import KFold

data = pd.read_csv("ecommerceDataset.csv", names=["category_name", "product_description"], header=None)
data.dropna(inplace=True)

data.category_name.replace("Clothing & Accessories", "Clothing_Accessories", inplace=True)
data['labelled_category'] = '__label__' + data['category_name'].astype(str)
data['labelled_text'] = data['labelled_category'] + ' ' + data['product_description']

def preprocess_text(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()

data['labelled_text'] = data['labelled_text'].map(preprocess_text)

kf = KFold(n_splits=3, shuffle=True, random_state=42)

params = {
    'epoch': 25,           # Number of epochs for training
    'lr': 0.1,             # Learning rate
    'dim': 50,             # Dimension of the word vectors
    'wordNgrams': 2,       # Use unigrams and bigrams for n-grams
    'bucket': 200000,      # Hash table size for subword information
    'minCount': 1,         # Minimum frequency of words to consider
    'loss': 'softmax'     # Type of loss function (use softmax for multi-class classification),
}


precisions = []
recalls = []

# Zmienna do przechowywania błędu w zależności od epok
epoch_errors = []
 
# Liczba epok do trenowania
epochs = 25
 
# Trening i śledzenie błędu dla każdej epoki
for epoch in range(1, epochs + 1):
    params['epoch'] = epoch  # Aktualizacja liczby epok w parametrach
    fold_errors = []  # Błędy dla wszystkich foldów w tej epoce
 
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        fold_train_data = data.iloc[train_index]
        fold_test_data = data.iloc[test_index]
 
        fold_train_data.to_csv(f"ecommerce_fold{fold+1}.train", columns=["labelled_text"], index=False, header=False)
        fold_test_data.to_csv(f"ecommerce_fold{fold+1}.test", columns=["labelled_text"], index=False, header=False)
 
        # Trening modelu FastText
        model = fasttext.train_supervised(input=f"ecommerce_fold{fold+1}.train", **params)
        
        # Testowanie modelu i zapisanie błędu (1 - accuracy)
        results = model.test(f"ecommerce_fold{fold+1}.test")
        error = 1 - results[2]  # Obliczenie błędu
        fold_errors.append(error)
 
    # Średni błąd dla tej epoki
    avg_error = sum(fold_errors) / len(fold_errors)
    epoch_errors.append(avg_error)
    print(f"Epoka {epoch}: Średni błąd {avg_error:.4f}")
 
# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), epoch_errors, marker='o', label='Błąd uczenia')
plt.title("Błąd uczenia w zależności od epoki")
plt.xlabel("Liczba epok")
plt.ylabel("Błąd (1 - dokładność)")
plt.xticks(range(1, epochs + 1))
plt.grid()
plt.legend()
plt.show()
 
import fasttext
import pandas as pd
import re
from sklearn.model_selection import KFold

# Wczytanie danych
data = pd.read_csv("ecommerceDataset.csv", names=["category_name", "product_description"], header=None)
data.dropna(inplace=True)

# Preprocessing
data.category_name.replace("Clothing & Accessories", "Clothing_Accessories", inplace=True)
data['labelled_category'] = '__label__' + data['category_name'].astype(str)
data['labelled_text'] = data['labelled_category'] + ' ' + data['product_description']

def preprocess_text(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()

data['labelled_text'] = data['labelled_text'].map(preprocess_text)

# Inicjalizacja 3-krotnej walidacji krzyżowej
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Parametry dla modelu FastText
params = {
    'epoch': 25,           # Number of epochs for training
    'lr': 0.1,             # Learning rate
    'dim': 50,             # Dimension of the word vectors
    'wordNgrams': 2,       # Use unigrams and bigrams for n-grams
    'bucket': 200000,      # Hash table size for subword information
    'minCount': 1,         # Minimum frequency of words to consider
    'loss': 'softmax'      # Type of loss function (use softmax for multi-class classification)
}

# Zmienna do przechowywania wyników
accuracies = []

# Walidacja krzyżowa
for fold, (train_index, test_index) in enumerate(kf.split(data)):
    # Podział danych na zbiór treningowy i testowy dla danego foldu
    fold_train_data = data.iloc[train_index]
    fold_test_data = data.iloc[test_index]
    
    # Zapisanie danych do plików dla danego foldu
    fold_train_data.to_csv(f"ecommerce_fold{fold+1}.train", columns=["labelled_text"], index=False, header=False)
    fold_test_data.to_csv(f"ecommerce_fold{fold+1}.test", columns=["labelled_text"], index=False, header=False)
    
    # Trening modelu FastText
    model = fasttext.train_supervised(input=f"ecommerce_fold{fold+1}.train", **params)
    
    # Testowanie modelu
    results = model.test(f"ecommerce_fold{fold+1}.test")
    
    # Wyświetlenie wyników dla danego foldu
    print(f"Fold {fold+1}:")
    print(f"Liczba poprawnych predykcji: {results[0]}")
    print(f"Liczba próbek: {results[1]}")
    print(f"Średnia dokładność (accuracy): {results[2]}")
    
    # Przechowywanie dokładności dla każdego foldu
    accuracies.append(results[2])

# Średnia dokładność po wszystkich foldach
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Średnia dokładność po wszystkich foldach: {average_accuracy}")

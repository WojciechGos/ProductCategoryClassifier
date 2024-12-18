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
    'lr': 0.2,             # Współczynnik uczenia
    'dim': 300,            # Ustalony wymiar wektorów słów
    'wordNgrams': 2,       # Ustalony wordNgrams (bigrams)
    'bucket': 200000,      # Rozmiar tabeli haszującej dla sub-słow
    'loss': 'ns',          # Funkcja straty: ns
    'minCount': 2,         # Zmieniony minCount
}

epoch_values = [5, 10, 25, 50, 100]

# Zmienna do zapisywania wyników
results_list = []

# Pętla przez różne wartości liczby epok
for epoch in epoch_values:
    params['epoch'] = epoch  # Zmiana liczby epok
    
    precisions = []
    recalls = []
    correct_predictions = []  # Lista na poprawnie zaklasyfikowane rekordy

    # Pętla przez zbiory KFold
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        fold_train_data = data.iloc[train_index]
        fold_test_data = data.iloc[test_index]

        # Zapisanie danych do formatu odpowiedniego dla fastText
        fold_train_data.to_csv("ecommerce_fold{}.train".format(fold+1), columns=["labelled_text"], index=False, header=False)
        fold_test_data.to_csv("ecommerce_fold{}.test".format(fold+1), columns=["labelled_text"], index=False, header=False)

        # Trening modelu
        model = fasttext.train_supervised(input="ecommerce_fold{}.train".format(fold+1), **params)

        # Testowanie modelu
        results = model.test("ecommerce_fold{}.test".format(fold+1), k=1)

        # Zbieranie poprawnie zaklasyfikowanych rekordów
        correct_predictions.append(results[0])  # Zapisywanie liczby poprawnych przewidywań dla każdego folda

        # Wyświetlanie wyników
        print("\n\n\n result:")
        print(results)

        print("\nFold {}:".format(fold+1))
        print("number of correct predicts: {}".format(results[0]))
        print("precision: {}".format(results[1]))
        print("recall: {}".format(results[2]))

        precisions.append(results[1])
        recalls.append(results[2])

    # Obliczenie średnich precyzji i recall dla tej liczby epok
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_correct_predictions = sum(correct_predictions) / len(correct_predictions)

    results_list.append({
        'epoch': epoch,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'correct_predictions': avg_correct_predictions 
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('fasttext_epoch_comparison.csv', index=False)

print("\n\nFinal Results saved to 'fasttext_epoch_comparison.csv'")

from sklearn.model_selection import KFold
import pandas as pd
import re

# Wczytanie danych
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

# Inicjalizacja 3-krotnej walidacji krzyżowej
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Utworzenie folderów dla każdego foldu
for fold, (train_index, test_index) in enumerate(kf.split(data)):
    # Podział danych na zbiór treningowy i testowy dla danego foldu
    fold_train_data = data.iloc[train_index]
    fold_test_data = data.iloc[test_index]
    
    # Zapisanie danych do plików dla danego foldu
    fold_train_data.to_csv(f"ecommerce_fold{fold+1}.train", 
                          columns=["labelled_text"], 
                          index=False, 
                          header=False)
    fold_test_data.to_csv(f"ecommerce_fold{fold+1}.test", 
                         columns=["labelled_text"], 
                         index=False, 
                         header=False)
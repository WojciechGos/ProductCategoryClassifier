# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import re


# # Wczytanie danych
# data = pd.read_csv("ecommerceDataset.csv", names=["category_name", "product_description"], header=None)
# data.dropna(inplace=True)
# data.category_name.replace("Clothing & Accessories", "Clothing_Accessories", inplace=True)
# data['labelled_category'] = '__label__' + data['category_name'].astype(str)
# data['labelled_text'] = data['labelled_category'] + ' ' + data['product_description']

# # def preprocess_text(text):
# #     text = re.sub(r'[^\w\s\']', ' ', text)
# #     text = re.sub(' +', ' ', text)
# #     return text.strip().lower()

# # data['labelled_text'] = data['labelled_text'].map(preprocess_text)

# # # 1. Wizualizacja rozkładu kategorii
# # plt.figure(figsize=(10, 6))
# # category_counts = data['category_name'].value_counts()
# # sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
# # plt.xlabel('Category')
# # plt.ylabel('Number of Samples')
# # plt.title('Category Distribution in the Dataset')
# # plt.xticks(rotation=45, ha='right')
# # plt.tight_layout()
# # plt.show()

# # # 2. Wizualizacja rozkładu długości opisów produktów
# # data['description_length'] = data['product_description'].apply(len)

# # plt.figure(figsize=(10, 6))
# # sns.histplot(data['description_length'], kde=True, color='blue', bins=30)
# # plt.xlabel('Length of Product Descriptions')
# # plt.ylabel('Frequency')
# # plt.title('Distribution of Product Description Lengths')
# # plt.show()

# # # 3. Wizualizacja najczęściej występujących słów w opisach
# # from sklearn.feature_extraction.text import CountVectorizer

# # # Utworzenie wektora cech (bag of words)
# # vectorizer = CountVectorizer(stop_words='english', max_features=20)
# # X = vectorizer.fit_transform(data['product_description'])
# # words = vectorizer.get_feature_names_out()

# # # Sumowanie wystąpień słów w opisach
# # word_counts = X.toarray().sum(axis=0)
# # word_freq = dict(zip(words, word_counts))

# # # Wizualizacja najczęściej występujących słów
# # plt.figure(figsize=(12, 6))
# # sns.barplot(x=list(word_freq.keys()), y=list(word_freq.values()), palette='Blues_d')
# # plt.xlabel('Words')
# # plt.ylabel('Frequency')
# # plt.title('Top 20 Most Frequent Words in Product Descriptions')
# # plt.xticks(rotation=45, ha='right')
# # plt.tight_layout()
# # plt.show()


# # import matplotlib.pyplot as plt

# # Lista plików z danymi
# train_files = ["ecommerce_fold1.train", "ecommerce_fold2.train", "ecommerce_fold3.train"]
# test_files = ["ecommerce_fold1.test", "ecommerce_fold2.test", "ecommerce_fold3.test"]

# # Listy do przechowywania liczby próbek w zbiorach treningowych i testowych
# train_sizes = []
# test_sizes = []

# # Wczytywanie danych i zliczanie liczby wierszy w każdym pliku
# for train_file, test_file in zip(train_files, test_files):
#     with open(train_file, 'r') as f:
#         train_sizes.append(sum(1 for _ in f))  # Zliczanie linii w pliku treningowym
#     with open(test_file, 'r') as f:
#         test_sizes.append(sum(1 for _ in f))  # Zliczanie linii w pliku testowym

# # Tworzenie wykresu słupkowego
# fold_labels = [f'Fold {i+1}' for i in range(len(train_sizes))]

# plt.figure(figsize=(10, 6))
# width = 0.4  # Szerokość słupków

# # # Rysowanie słupków dla zbiorów treningowych i testowych
# # plt.bar([x - width/2 for x in range(1, len(train_sizes) + 1)], train_sizes, width=width, label='Train Set', color='skyblue')
# # plt.bar([x + width/2 for x in range(1, len(test_sizes) + 1)], test_sizes, width=width, label='Test Set', color='salmon')

# # # Dodawanie etykiet i tytułów
# # plt.xlabel('Fold')
# # plt.ylabel('Number of Samples')
# # plt.title('Number of Samples in Train and Test Sets for Each Fold')
# # plt.xticks(range(1, len(fold_labels) + 1), fold_labels)
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# train_files = ["ecommerce_fold1.train", "ecommerce_fold2.train", "ecommerce_fold3.train"]
# test_files = ["ecommerce_fold1.test", "ecommerce_fold2.test", "ecommerce_fold3.test"]

# # Słowniki do przechowywania liczebności każdej klasy w zbiorach treningowych i testowych dla każdego folda
# train_class_counts = {}
# test_class_counts = {}

# # Wczytywanie danych i zliczanie liczebności klas dla każdego folda
# for fold_num, (train_file, test_file) in enumerate(zip(train_files, test_files), start=1):
#     # Wczytanie danych
#     train_data = pd.read_csv(train_file, header=None, names=["labelled_text"])
#     test_data = pd.read_csv(test_file, header=None, names=["labelled_text"])
    
#     # Wyodrębnienie kategorii z kolumny `labelled_text`
#     train_data['category'] = train_data['labelled_text'].str.extract(r'__label__(\w+)')
#     test_data['category'] = test_data['labelled_text'].str.extract(r'__label__(\w+)')
    
#     # Zliczanie liczby rekordów każdej klasy (kategorii)
#     train_class_counts[f'Fold {fold_num} Train'] = train_data['category'].value_counts()
#     test_class_counts[f'Fold {fold_num} Test'] = test_data['category'].value_counts()

# # Tworzenie DataFrame z liczebnością klas w każdym foldzie
# train_class_counts_df = pd.DataFrame(train_class_counts).fillna(0).sort_index()
# test_class_counts_df = pd.DataFrame(test_class_counts).fillna(0).sort_index()

# # Tworzenie wykresu słupkowego dla zbiorów treningowych i testowych
# fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# # Wykres dla zbiorów treningowych
# train_class_counts_df.plot(kind='bar', ax=axes[0], color=plt.cm.Paired.colors)
# axes[0].set_title('Class Distribution in Training Sets for Each Fold')
# axes[0].set_ylabel('Number of Samples')
# axes[0].legend(title='Fold', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Wykres dla zbiorów testowych
# test_class_counts_df.plot(kind='bar', ax=axes[1], color=plt.cm.Paired.colors)
# axes[1].set_title('Class Distribution in Test Sets for Each Fold')
# axes[1].set_xlabel('Class (Category)')
# axes[1].set_ylabel('Number of Samples')
# axes[1].legend(title='Fold', bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
train_files = ["ecommerce_fold1.train", "ecommerce_fold2.train", "ecommerce_fold3.train"]
test_files = ["ecommerce_fold1.test", "ecommerce_fold2.test", "ecommerce_fold3.test"]

# Słownik do przechowywania liczebności klas dla każdego pliku
class_counts = {}

# Wczytywanie danych i zliczanie liczebności klas dla każdego pliku
for file in train_files + test_files:
    # Wczytanie danych
    data = pd.read_csv(file, header=None, names=["labelled_text"])
    
    # Wyodrębnienie kategorii z kolumny `labelled_text`
    data['category'] = data['labelled_text'].str.extract(r'__label__(\w+)')
    
    # Zliczanie liczby rekordów dla każdej klasy
    class_counts[file] = data['category'].value_counts()

# Tworzenie DataFrame z liczebnością klas dla każdego pliku, uzupełnienie brakujących wartości zerami
class_counts_df = pd.DataFrame(class_counts).fillna(0).sort_index()

# Tworzenie wykresu słupkowego
fig, ax = plt.subplots(figsize=(12, 6))
class_counts_df.T.plot(kind='bar', ax=ax, width=0.8)

# Dodanie etykiet i tytułów
ax.set_title('Liczebność każdej klasy w poszczególnych plikach')
ax.set_xlabel('Plik')
ax.set_ylabel('Liczba Przykładów')
ax.legend(title='Klasa', loc='upper left', bbox_to_anchor=(1, 1))

# Dodanie wartości liczbowych nad słupkami
for container in ax.containers:
    ax.bar_label(container, label_type='edge')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
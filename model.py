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
    'epoch': 25,           # Number of epochs for training  // 5 15 25 45
    'lr': 0.1,             # Learning rate
    'dim': 50,             # Dimension of the word vectors
    'wordNgrams': 2,       # Use unigrams and bigrams for n-grams
    'bucket': 200000,      # Hash table size for subword information
    'minCount': 1,         # Minimum frequency of words to consider
    'loss': 'softmax'     # Type of loss function (use softmax for multi-class classification),
}


precisions = []
recalls = []

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    fold_train_data = data.iloc[train_index]
    fold_test_data = data.iloc[test_index]
    
    fold_train_data.to_csv("ecommerce_fold{}.train".format(fold+1), columns=["labelled_text"], index=False, header=False)
    fold_test_data.to_csv("ecommerce_fold{}.test".format(fold+1), columns=["labelled_text"], index=False, header=False)    
    model = fasttext.train_supervised(input="ecommerce_fold{}.train".format(fold+1), **params)

    results = model.test("ecommerce_fold{}.test".format(fold+1), k=1)
    model_save_path = f"model{fold+1}.bin"
    model.save_model(model_save_path)

    print("\n\n\n result:")
    print(results)
    

    print("\nFold {}:".format(fold+1))
    print("number of correct predicts: {}".format(results[0]))
    print("precision: {}".format(results[1]))
    print("recall: {}".format(results[2]))
    precisions.append(results[1])
    recalls.append(results[2])

avg_precision = sum(precisions) / len(precisions)
avg_recalls = sum(recalls) / len(recalls)

print("avarage precission (all folds): {}".format(avg_precision))
print("avarage recall (all folds): {}".format(avg_recalls))

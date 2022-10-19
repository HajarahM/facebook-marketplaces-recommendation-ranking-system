import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# Actual data to train
# import data
# get training set
root_dir: str = 'Products.csv'
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"The file {root_dir} does not exist")
products = pd.read_csv(root_dir, lineterminator='\n')
print(products)

# # load categories from training set
# categories = [
#     train['product_name'],
#     train['category'],
#     train['product_description'],
#     train['location']
# ]

# instantiate the vectorizer object
countvectorizer = CountVectorizer()
tfidfvectorizer = TfidfVectorizer()

# convert the documents into a matrix
count_matrix = countvectorizer.fit_transform(products['product_description'])
tfidf_matrix = tfidfvectorizer.fit_transform(products['product_description'])

countvect = count_matrix.toarray()
tfidfvect = tfidf_matrix.toarray()

print("Count Vectorizer\n")
print(countvect)
print("\nTD-IDF Vectorizer\n")
print(tfidfvect)

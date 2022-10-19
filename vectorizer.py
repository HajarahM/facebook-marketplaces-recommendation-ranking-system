import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# import data
# get training set
root_dir: str = 'Products.csv'
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"The file {root_dir} does not exist")
products = pd.read_csv(root_dir, lineterminator='\n')

# load categories from training set
categories = [
    products['product_name'],
    products['category'],
    products['product_description'],
    products['location']
]

# instantiate the vectorizer object
countVectorizer = CountVectorizer(analyzer='word', stop_words='english')
ifidfvectorizer = TfidfVectorizer(analyzer='word',stop_words='english')

# convert the documents into a matrix


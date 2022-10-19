from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from scipy import sparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Actual data to train
# import data
# get training set
root_dir: str = 'cleaned_products.csv'
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"The file {root_dir} does not exist")
products = pd.read_csv(root_dir, lineterminator='\n')
print(products.info())

# load categories from training set
product_name = products['product_name']
product_category = products['category']
product_description = products['product_description']
product_location = products['location']


# instantiate the vectorizer object
countvectorizer = CountVectorizer()
tfidfvectorizer = TfidfVectorizer()

# convert the documents into a matrix
# Count
count_productname = countvectorizer.fit_transform(product_name).toarray()
count_productcategory = countvectorizer.fit_transform(product_category).toarray()
count_productdescription = countvectorizer.fit_transform(product_description).toarray()
count_productlocation = countvectorizer.fit_transform(product_location).toarray()

# Vectorize and get TF-IDF scores
response_prouductname = tfidfvectorizer.fit_transform(product_name).toarray()
response_prouductcategory = tfidfvectorizer.fit_transform(product_category).toarray()
response_prouductdescription = tfidfvectorizer.fit_transform(product_description).toarray()
response_prouductlocation = tfidfvectorizer.fit_transform(product_location).toarray()

responses_concat = np.concatenate((response_prouductname, response_prouductcategory, response_prouductdescription, response_prouductlocation), axis=1)
sparse_responses = sparse.csr_matrix(responses_concat)
print(responses_concat.shape)

training_outputs = products['price']
# Regres
model = linear_model.LinearRegression()
model.fit(sparse_responses, training_outputs)
print(model.coef_)
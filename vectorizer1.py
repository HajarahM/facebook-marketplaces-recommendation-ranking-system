import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# # Example train data
# # set of documents
# train = ['The sky is blue.','The sun is bright.']
# test = ['The sun in the sky is bright', 'We can see the shining sun, the bright sun.']

# Actual data to train
# import data
# get training set
root_dir: str = 'Products.csv'
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"The file {root_dir} does not exist")
train = pd.read_csv(root_dir, lineterminator='\n')

# load categories from training set
categories = [
    train['product_name'],
    train['category'],
    train['product_description'],
    train['location']
]

# instantiate the vectorizer object
countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

# convert the documents into a matrix
count_wm = countvectorizer.fit_transform(train['product_description'])
tfidf_wm = tfidfvectorizer.fit_transform(train['product_description'])

# retrieve the terms found using get_feature_names() method
count_tokens = countvectorizer.get_feature_names_out()
tfidf_tokens = tfidfvectorizer.get_feature_names_out()

df_countvect = pd.DataFrame(data = count_wm.toarray(), columns=count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns=tfidf_tokens)

print("Count Vectorizer\n")
print(df_countvect)
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)

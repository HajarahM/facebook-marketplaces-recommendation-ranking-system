import pandas as pd
import numpy as np

df_images = pd.read_csv('Images.csv', index_col=0, lineterminator='\n')
print(df_images)

#convert colomn datatypes
df_products = pd.read_csv("Products.csv", index_col=0, lineterminator='\n')
print(df_products.info())

#drop Rows with missing value / NaN in any column

clean_df_products = df_products.dropna()
print("Modified Dataframe : ")
print(clean_df_products)

#price
df_products['price'] = df_products['price'].replace('[\£,]', '', regex=True).astype(float)
df_products['price'] = pd.to_numeric(df_products['price'])
#location
df_products['location'] = df_products['location'].astype('category')

print(df_products)
df_products['location'].describe()




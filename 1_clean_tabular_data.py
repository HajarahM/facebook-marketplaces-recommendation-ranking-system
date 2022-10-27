import pandas as pd

#import csv files as dataframes
products_df = pd.read_csv('Products.csv', lineterminator='\n')
images_df = pd.read_csv('Images.csv', lineterminator='\n')

#rename 'id' columns
df = products_df.rename(columns={'id':'product_id'})
idf = images_df.rename(columns={'id':'image_id'})

#merge the 2 tables
combined_df = pd.merge(df, idf, how="inner", on=["product_id"])
#delete unnamed columns (index columns from initial tables)
combined_df.drop(combined_df.columns[combined_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#change datatype of price column to float, remove currency sign, and make numeric
combined_df['price'] = combined_df['price'].replace('[\£,]', '', regex=True).astype(float)
combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')

#clean product names - strip all text after first "|"
combined_df['product_name'] = combined_df['product_name'].str.split('|').str[0]

#split category column into main and sub categories
combined_df['main_category'], combined_df['sub_category'] = combined_df['category'].str.split('/',1).str

#Specify Category columns
combined_df['category'] = combined_df['category'].astype('category')
combined_df['location'] = combined_df['location'].astype('category')

#delete rows with empty data, missing values
cleaned_df = combined_df.dropna()

#print statements
print(f'Products dataset: {len(df)}')
print(f'Image dataset {len(idf)}')
print(f'Combined products dataframe: {len(combined_df)}')
print(f'Cleaned products dataframe: {len(cleaned_df)} some products have mulitple images')
print(cleaned_df.head())

# save cleaned_products dataframe to csv file
cleaned_df.to_csv('cleaned_products.csv')




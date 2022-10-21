# Import Python libraries for data manipulation and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

# Import the Python machine learning libraries we need
# Sklearn processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
# Sklearn classification algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# Sklearn classification model evaluation function
from sklearn.metrics import accuracy_score

# Import some convenient functions.  
# from functions import *

# The Process - 1. Define the task, 2. Acquire Clean Data, 3. Understand the Data, 4. Prepare Data, 5. Build Models, 6. Evaluate Models, 7. Finalize & Deploy
# 1. Define the Task: The Task is to Make predictions about the price of specific product from a set of matrix from facebook marketplace
# 2. Acquire Clean Data
# Load the Dataset
dataset = pd.read_csv("Products.csv", lineterminator='\n')
# Inspect Data: Identify the number of features (colomns) and samples (rows)




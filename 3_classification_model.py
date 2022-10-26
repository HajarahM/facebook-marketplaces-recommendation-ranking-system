import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

products_train = pd.read_csv('cleaned_products.csv', sep=',',header=0, lineterminator='\n')
products_test = pd.read_csv('cleaned_products.csv', sep=',',header=0, lineterminator='\n')

ohe = preprocessing.OneHotEncoder()
products_train = products_train.values.reshape (1,-1)
products_train = ohe.fit_transform(products_train)
# for product_name in products_train:
#     if products_train[product_name].dtype == object:
#         products_train[product_name] = ohe.fit_transform(products_train[product_name])
#     else:
#         pass

y_tr = products_train[:,5:]
X_tr = products_train[:,2:]

y_test = products_test.iloc[:,5]
X_test = products_test.iloc[:,2:]

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, y_tr)
LR.predict(X_test)
round(LR.score(X_test,y_test), 4)

SVM = svm.SVC(decision_function_shape="ovo").fit(X_tr, y_tr)
SVM.predict(X_test)
round(SVM.score(X_test, y_test), 4)

RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_tr, y_tr)
RF.predict(X_test)
round(RF.score(X_test, y_test), 4)

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_tr, y_tr)
NN.predict(X_test)
round(NN.score(X_test, y_test), 4)
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


iowa_file_path = 'train.csv'

# Fill in the line below to read the file into a variable home_data
df = pd.read_csv(iowa_file_path)


#Target of prediction:
y = df["SalePrice"]

#Features:
X = df[["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]]


#split the data:
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#We set the max_leaf_nodes property within a function so that we can call it multiple times until 
#we are in the best zone between over and underfitting our model:
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

#We call the function multiple times with different values for max_leaf_nodes so as to compare the accuracy of models:
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
""" OUTPUT: 
Max leaf nodes: 5                Mean Absolute Error:  35190
Max leaf nodes: 50               Mean Absolute Error:  27825    SMALLEST SO 50 NODES ARE OPTIMAL
Max leaf nodes: 500              Mean Absolute Error:  32662
Max leaf nodes: 5000             Mean Absolute Error:  33382"""

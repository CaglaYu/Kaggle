import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

#This time we do not built a decision tree model but a random forest model instead:
model = RandomForestRegressor(random_state=1)
model.fit(train_X,train_y)
predictions= model.predict(val_X)
print(f"MAE = {mean_absolute_error(val_y, predictions)}")
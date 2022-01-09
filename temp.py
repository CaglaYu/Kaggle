import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


iowa_file_path = 'train.csv'

# Fill in the line below to read the file into a variable home_data
df = pd.read_csv(iowa_file_path)
print(df.describe())
print(df.columns)

#Target of prediction:
y = df["SalePrice"]

#Features:
X = df[["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]]

#define the model:
my_model = DecisionTreeRegressor(random_state=1)
my_model.fit(X, y)
print(X.head())
my_predictions = my_model.predict(X)
print(mean_absolute_error(y, my_predictions))



#redfine model with evaluation using splitting:
#Until now, we have evaluated the model with the data which we used to form the model in the first placr
#Now we split the data and use one part to form a model and the other part to evaluate the model:
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
my_model = DecisionTreeRegressor()
my_model.fit(train_X, train_y)
my_predictions = my_model.predict(val_X)
print(mean_absolute_error(val_y, my_predictions))


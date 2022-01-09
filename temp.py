import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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
print(my_model.predict(X.head()))

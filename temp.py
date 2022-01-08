import pandas as pd

iowa_file_path = 'train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)
display(home_data.describe())

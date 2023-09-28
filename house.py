import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle




df = pd.read_csv("E:\\Machine_learning_Project\\Machine_learning_Project\\house_data.csv")




X = df.iloc[:, 3:]
y = df.iloc[:,2]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))

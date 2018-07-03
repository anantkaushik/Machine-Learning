# Importing the required libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

#Encoding categorial dataLabel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the Dummy Variable Trap
x = x[:,1:]

# Splitting the data set into the Training set and the Test Set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results.
y_pred = regressor.predict(x_test)

# Optimal Solution 
# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

# Splitting the data set into the Training set and the Test Set
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor1 = LinearRegression()
regressor1.fit(x_train1, y_train1)

# Predicting the Test set results.
y_pred1 = regressor1.predict(x_test1)
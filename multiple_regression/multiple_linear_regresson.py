#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
dataset =  pd.read_csv('50_Startups.csv')

# Seperate X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#label encoding and One hot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X = X[:, 1:]

#Spliting the Traing and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)

#Fit the regression model with our data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predictin for test cases
y_pred = regressor.predict(X_test)


#import statsmodels.formula.api as sm
import statsmodels.api as sm
X =  sm.add_constant(X)
#X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Remove X2 Categorical Variable and then Perfrom Regression again

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
####Right Code to Visualize The regression line between R&D Spend and Profit
#### Right Your Code here

x_opt_train, x_opt_test, Y_opt_train, Y_opt_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)


# Linear Regression Model fitting the dataset
regressor1 = LinearRegression()
regressor1.fit(x_opt_train, Y_opt_train)


# Predict the test samples
Y_opt_pred = regressor1.predict(x_opt_test)


plt.scatter(X_train[:,3], y_train, color = 'red')
plt.plot(X_train[:,3], regressor.predict(X_train), color = 'blue')
plt.title('R&D vs Profit (Training set[Multifeature])')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()

plt.scatter(X_test[:,3], y_test, color = 'red')
plt.plot(X_test[:,3], regressor.predict(X_test), color = 'blue')
plt.title('R&D vs Profit (Testing set[Multifeature])')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()


plt.scatter(x_opt_train[:,1], Y_opt_train, color = 'red')
plt.plot(x_opt_train[:,1], regressor1.predict(x_opt_train), color = 'blue')
plt.title('R&D vs Profit (Training set[R&D])')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()

plt.scatter(x_opt_test[:,1], Y_opt_test, color = 'red')
plt.plot(x_opt_test[:,1], Y_opt_pred, color = 'blue')
plt.plot(x_opt_train[:,1], regressor1.predict(x_opt_train), color = 'green')
plt.title('R&D vs Profit (Testing set[R&D])')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()
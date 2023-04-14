import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load data
cardf = pd.read_csv('C:/Users/hsri2/OneDrive/Desktop/Data sheet/CarPrice.csv')
cardf.head()

# Remove car ID column
del cardf['car_ID']

# Check data info and missing values
cardf.info()
cardf.isnull().sum()

# Encode categorical variables
cat_cols = cardf.select_dtypes(include='object').columns
num_cols = cardf.select_dtypes(exclude='object').columns

LD = LabelEncoder()
for i in cat_cols:
    cardf[i] = LD.fit_transform(cardf[i])

# Scale numerical variables
MMS = MinMaxScaler()
cardf[num_cols] = MMS.fit_transform(cardf[num_cols])
cardf.head()

# Split into training and test sets
x = cardf.drop(['price'], axis=1)
y = cardf['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=0)  

# Train linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

# Print coefficients and variance score
print('Coefficients: ', reg.coef_)
print('Variance score: {}'.format(reg.score(x_test, y_test)))

# Calculate and print RMSE
mse = mean_squared_error(y_test, y_pred)  
rmse = math.sqrt(mse)
print(rmse)

# Evaluate model using cross-validation
scores = cross_val_score(reg, x, y, cv=5)
print('Cross-validation scores:', scores)
print('Mean score:', scores.mean())

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# Plot residuals
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Plot distribution of residuals
sns.histplot(y_test - y_pred)
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")


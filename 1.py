import pandas as pd
df = pd.read_csv("data.csv")
print(df.head(30))
print(df.tail(30))

X = df[['study_hours','sleep_hours','attendance']]

y = df['grades']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error,r2_score
y_pred = model.predict(X_test)
print("intercept", model.intercept_)
print("coefficient", model.coef_)
print("mean squared error", mean_squared_error(y_test,y_pred))
print("r2 score", r2_score(y_test,y_pred))

import matplotlib.pyplot as plt
import numpy as np
plt.scatter(y_test, y_pred)

plt.xlabel("ACTUAL")
plt.ylabel("PREDICTED")
plt.title("Actual vs Predicted Grades")
plt.show()
from pkgutil import read_code
from statistics import linear_regression
from tkinter import Y
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt # pythons plotting package

def calculateResult(profitPercent):
    if profitPercent < 100:
        return 1
    elif profitPercent >= 100 and profitPercent <= 125:
        return 2
    elif profitPercent >= 125 and profitPercent <= 175:
        return 3
    elif profitPercent >= 175 and profitPercent <= 200:
        return 4
    else:
        return 5

df = pd.read_csv('movies.csv')

# Labelling Encoding Categorical Variables
label_encoder = LabelEncoder()
df['rating_label'] = label_encoder.fit_transform(df['rating'])
df['genre_label'] = label_encoder.fit_transform(df['genre'])
df['writer_label'] = label_encoder.fit_transform(df['writer'])
df['director_label'] = label_encoder.fit_transform(df['director'])
df['star_label'] = label_encoder.fit_transform(df['star'])
df['country_label'] = label_encoder.fit_transform(df['country'])
df['company_label'] = label_encoder.fit_transform(df['company'])


#Cleaning the Data
df = df.dropna()

#Calculating Profit Percentage
df['profitPercent'] = df['gross']*100/df['budget']

# Creating ouput Variable
df['result'] = df['profitPercent'].apply(lambda value: calculateResult(value))


# Creating Input DataSet
df_main = df[['rating_label','genre_label','score','director_label','writer_label','star_label','profitPercent','company_label','result']]


# Dividing dataset into Training, Validation and Test

train = df_main[:2700]
val = df_main[2700:4200]
test = df_main[4200:]


X_train = train.drop('result',1)
Y_train = train['result']

X_val = val.drop('result',1)
Y_val = val['result']

X_test = test.drop('result',1)
Y_test = test['result']

# Scaling of Values using Z-Score

X_train = (X_train - X_train.mean())/X_train.std()
Y_train = (Y_train - Y_train.mean())/Y_train.std()

X_val = (X_val - X_val.mean())/X_val.std()
Y_val = (Y_val - Y_val.mean())/Y_val.std()

X_test = (X_test - X_test.mean())/X_test.std()
Y_test = (Y_test - Y_test.mean())/Y_test.std()




#print(X_train.head(10))
#print(Y_train.head(10))

linear_regression = LinearRegression()
lr = linear_regression.fit(X_train,Y_train)

# print(mse(Y_train,lr.predict(X_train)))
# print(mse(Y_val,lr.predict(X_val)))
# print(mse(Y_test,lr.predict(X_test)))

# alphas = [0.01/2,0.02/2,0.03/2,0.04/2,0.05/2,0.075/2,0.1/2]

# mses = []

# for alpha in alphas:
#     lasso = Lasso(alpha=alpha)
#     lasso.fit(X_train,Y_train)
#     pred = lasso.predict(X_val)
#     mses.append(mse(Y_val,pred))
#     #print(mse(Y_val,pred))

# plt.plot(alphas,mses)
# plt.show()


alpha = 0.015/2
lasso = Lasso(alpha = alpha)
lasso.fit(X_train,Y_train)

print(mse(Y_val,lasso.predict(X_val)))
print(mse(Y_test,lasso.predict(X_test)))

print(lr.coef_)
print(lr.intercept_)






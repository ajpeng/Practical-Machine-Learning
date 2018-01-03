import pandas as pd
import quandl, math, datetime
#numpy is required since python doesnt have arrays
import numpy as np
#using SVM to show regression
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')

#Getting labels from quandl data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#High low percentage change
df['HL_PCT']= (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
#Percent change 
df['PCT_Change']= (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#predict forecast for the next 1% into the future
forecast_out= int(math.ceil(0.01*len(df)))
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)

#print(df.head())
#features are x , labels are y
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
#X = X[:-forecast_out+1]
#in practice, you are also required to scale the x data above 


df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

#print(len(X),len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y , test_size=0.2)
#clf = svm.SVR()
#comment out if you want to user Support Vector Machine
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# The squared error 
#print(accuracy)

#30 day accuracy
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.xlabel('Price')
plt.show()

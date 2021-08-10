import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from sklearn.model_selection import train_test_split

##README: this models runs perfectly well for all .csv files. All you need to do is (1.) clean the datasets,
## (2.) change dataset names when concatenating, (3.) changethe data you are removing and withholding to do the actual prediction,
## (4.) change the range length in the dummy indicator, (5.) figure out the data column(s) of your regressors and labels,
## (6.) change the input dimensions in the input layer of the model

#1. Import all data and clean data
discretionarydata=pd.read_csv('Consumerdiscretionary.csv')
discretionarydata1=pd.DataFrame(discretionarydata)
discretionclean=discretionarydata1.drop(columns=['fyearq', 'fqtr', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat','gsubind'])
#print(discretionclean)

discretionarydata2=pd.read_csv('Consumerstaples.csv')
discretionarydata3=pd.DataFrame(discretionarydata2)
consumerclean=discretionarydata3.drop(columns=['fyearq', 'fqtr', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat','gsubind'])
#print(consumerclean)

discretionarydata4=pd.read_csv('Industrials.csv')
discretionarydata5=pd.DataFrame(discretionarydata4)
industrialsclean=discretionarydata5.drop(columns=['fyearq', 'fqtr', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat','gsubind'])
#print(industrialsclean)

discretionarydata6=pd.read_csv('Energy.csv')
discretionarydata7=pd.DataFrame(discretionarydata6)
energyclean=discretionarydata7.drop(columns=['fyearq', 'fqtr', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat','gsubind'])
#print(energyclean)

discretionarydata8=pd.read_csv('Healthcare.csv')
discretionarydata9=pd.DataFrame(discretionarydata8)
healthcareclean=discretionarydata9.drop(columns=['fyearq', 'fqtr', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat','gsubind'])
#print(healthcareclean)

discretionarydata10=pd.read_csv('Financials.csv')
discretionarydata11=pd.DataFrame(discretionarydata10)
financialsclean=discretionarydata11.drop(columns=['fyearq', 'fqtr', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat','gsubind'])
#print(financialsclean)

discretionarydata12=pd.read_csv('Communicationservices.csv')
discretionarydata13=pd.DataFrame(discretionarydata12)
communicationclean=discretionarydata13.drop(columns=['fyearq', 'fqtr', 'indfmt', 'consol', 'popsrc', 'datafmt', 'curcdq', 'datacqtr', 'datafqtr', 'costat','gsubind'])
#print(communicationclean)


#2. Combine all dataframes into one mega dataframe)
megaset=pd.concat([discretionclean, consumerclean, industrialsclean, energyclean, healthcareclean, financialsclean, communicationclean])


#3. Start creating test data by isolating a certain range of dates
part1=megaset.loc[megaset['datadate'] == '3/31/2019']
part2=megaset.loc[megaset['datadate'] == '6/30/2019']
part3=megaset.loc[megaset['datadate'] == '9/30/2019']
part4=megaset.loc[megaset['datadate'] == '12/31/2019']
testx=pd.concat([part1,part2,part3,part4])
#Clean dataset a bit more
test1=testx.drop(['GVKEY'], axis=1)
test2=test1.drop(['Unnamed: 18'], axis= 1)
test3=test2.dropna()
test4=test3.set_index([pd.Index(list(range(1095)))])
a=range(1095)
b=np.array(a)
#Create dummy indicator for dates because original date format was in string form
df=pd.DataFrame(data=b, columns=['Indicator'])
test_data=pd.concat([test4, df], axis=1)
print(test_data)

#Start creating test data by dropping the previous range of dates
trainx=megaset.set_index('datadate')
train1=trainx.drop(['3/31/2019', '6/30/2019', '9/30/2019', '12/31/2019'], axis=0)
#Clean dataset a bit more
train2=train1.drop(['GVKEY'], axis=1)
train3=train2.drop(['Unnamed: 18'], axis=1)
train4=train3.dropna()
train5=train4.reset_index()
train6=train5.set_index([pd.Index(list(range(12339)))])
c=range(12339)
d=np.array(c)
#Create dummy indicator for dates because original date format was in string form
df1=pd.DataFrame(data=d, columns=['Indicator'])
train_data=pd.concat([train6, df1], axis=1)
print(train_data)

#4. Isolate indicators from labels in the training set and the test set
dates=train_data.index
X=train_data.iloc[:,6]
Y=train_data.iloc[:,4]
Xi_test=test_data.iloc[:,6]
yi_test=test_data.iloc[:,4]

#5. Model function
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True)
model = Sequential()
model.add(Dense(600,activation='relu',input_dim=1))

model.add(Dense(300,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(150,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(75,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(30,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1,activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=128)

#Results from model
loss = model.evaluate(X_test, y_test)
print('sqrt loss', np.sqrt(loss))
print('standard deviation', train_data['revtq'].std())

#Verify accuracy of results for training data
predictions = model.predict(X)

predictions_list = map(lambda x: x[0], predictions)
print('predlist', predictions_list)
predictions_series = pd.Series(predictions_list,index=dates)
dates_series = pd.Series(dates)
#for x in predictions:
 #   print('prediction', x[0])

#Use previously generated model to generate labels for the test data
Predicted_sales = model.predict(Xi_test)
new_dates_series=pd.Series(Xi_test.index)
new_predictions_list = map(lambda x: x[0], Predicted_sales)
new_predictions_series = pd.Series(new_predictions_list,index=new_dates_series)

#Export to CSV
new_predictions_series.to_csv("predicted_saless.csv",header=False)

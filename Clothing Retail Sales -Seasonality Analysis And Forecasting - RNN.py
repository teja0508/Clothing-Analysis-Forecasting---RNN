"""
Clothing Retail Sales -Seasonality Analysis & Forecasting With Recurrent Neural Networks -RNN :
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#By Using Parse_Date=True , pandas will automatically detect date column as datetime object :

df=pd.read_csv('RSCCASN.csv',parse_dates=True,index_col='DATE')

print(df.head())

print(df.info())


df.columns=['Sales']

sns.set_style('whitegrid')
df.plot(figsize=(12,8))
plt.show()


""" 
Determining Train / Test Split Index :
"""

print(len(df))

print(len(df)-18)

test_size=18

test_ind=len(df)-test_size

print(test_ind)

""" 
Here , For forecasting we should train our model for atleast a year 's cycle , in order to get it familiar with
the seasonality of sales..For simplicity , i will be taking 18 as test size , since 18 months = 1.5 years

and before that , rest of the data will be my training data :
"""


train=df.iloc[:test_ind]
test=df.iloc[test_ind:]


print(train)

print(test)


""" 
SCALING OF DATA :

"""


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

scaled_train=scaler.fit_transform(train)
scaled_test=scaler.transform(test)


""" 
TimeSeriesGenerator :


For validation test set generator my batch size must be less than test size , i.e. , 18 , in order
to run properly without error. I would be taking my length of batch as 12 , because it is one whole 
year..
"""

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

length=12

generator=TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=1)

X,y=generator[0]

print(X)
print(y)
print(scaled_train[13])


""" 
CREATING MODEL:


"""

from tensorflow.keras.layers import Dense,SimpleRNN,LSTM
from tensorflow.keras.models import Sequential



n_features=1     #Sales column

model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(length,n_features)))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')


print(model.summary())


from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_loss',patience=2)

validation_gen=TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=1)
model.fit_generator(generator,epochs=20,validation_data=validation_gen,callbacks=[early_stop])

losses=pd.DataFrame(model.history.history)
losses.plot()
plt.show()



test_predictions=[]

first_eval_batch=scaled_train[-length:]
current_batch=first_eval_batch.reshape(1,length,n_features)

for i in range(len(test)):
    current_pred=model.predict(current_batch)[0]

    test_predictions.append(current_pred)

    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)



true_pred=scaler.inverse_transform(test_predictions)

df2=test.copy()
df2['Predictions']=true_pred

print(df2)

df2.plot(figsize=(12,8))
plt.title('Sales VS Predicitons')
plt.show()


""" 
FORECASTING VALUES INTO UNKNOWN FUTURE By USING FULL DATASET AS TRAINING:
"""

full_Scaler=MinMaxScaler()

full_scaled_data=full_Scaler.fit_transform(df)  #Transforming Full Data :

length=12

generator=TimeseriesGenerator(full_scaled_data,full_scaled_data,length=length,batch_size=1)

model=Sequential()

model.add(LSTM(100,activation='relu',input_shape=(length,n_features)))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')

model.fit_generator(generator,epochs=8)


"""
periods is the number of months we want to predict , so here i took 12 , as i want to predict
next 12 months forecast.You can take any number you want .. but do remember , longer the length
of period , more will be the noisy data....



SO it would be much better if you choose the same length for periods as 
the length of your test size , i.e. , 12..
"""
forecast=[]
periods=12

first_eval_batch=scaled_train[-length:]
current_batch=first_eval_batch.reshape(1,length,n_features)

for i in range(periods):
    current_pred=model.predict(current_batch)[0]

    forecast.append(current_pred)

    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
    
forecast_pred=full_Scaler.inverse_transform(forecast)

print(df)
print(forecast_pred)

#You can see that forecast_pred values are actually values predicted after the dataframe's values


""" 
Now let us create a proper dataframe with comparisons..
we have to create new timestamp range for future predictions
"""

""" 

pd.date_Range will create automatically new dates with intervals , start parameter will take the 
starting point , one after our original dataframe , periods is intervals , i.e., 12[already defined 
into periods variable], freq=offset aliases MS -> monthly sequence intervals


"""
forecast_index=pd.date_range(start='2019-11-01',periods=periods,freq='MS')

print(forecast_index)

forecast_Df=pd.DataFrame(data=forecast_pred,index=forecast_index,columns=['Forecast'])

print(forecast_Df)


ax=df.plot()
forecast_Df.plot(ax=ax)
plt.title("Forecasting Sales Prediction Into Unknown Future")
plt.show()

ax=df.plot()
forecast_Df.plot(ax=ax)
plt.xlim('2018-01-01','2020-12-01')  #Zoom our graph into mentioned X - AXIS Values
plt.title("Forecasting Sales Prediction Into Unknown Future between 2018 to 2020")
plt.show()
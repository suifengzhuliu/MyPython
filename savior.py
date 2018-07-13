from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
dataset = read_csv('savior.csv', header=0, index_col=0)
values = dataset.values

print('first values top 10  is ')
print(len(values), values[:10, :])

# 输入的天数，即根据in_step天的数据来预测下一天的数据
in_step = 21
# integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')


print('values top 10  is ')
print(len(values), values[:10, :])




n_train_hours = 280
print("用于测试的数据")
size = 14
oriTestData = values[n_train_hours + in_step:n_train_hours + in_step + size, 0]
print(oriTestData)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(values, in_step, 1)

# print("before reframed.head")
# print(reframed.values[:10, :8])

# drop columns we don't want to predict
a = 4

reframed.drop(reframed.columns[[ in_step * a + 0,in_step * a + 1,in_step * a + 2]], axis=1, inplace=True)
# a = 8
# for i in range(59, 29, -1):
#     # print i*a+1,i*a+2,i*a+3,i*a+4,i*a+5,i*a+6,i*a+7
#     reframed.drop(reframed.columns[[i * a + 7, i * a + 6, i * a + 5, i * a + 4, i * a + 3, i * a + 2, i * a + 1]], axis=1, inplace=True)
# print("after reframed.head  ,tail 10")
# print(reframed.values[-10:, :8])
# 
# print("values   reframed shape")
# print(reframed.shape)




# split into train and test sets
# values = reframed.values
values = scaler.fit_transform(reframed.values)







train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=30, batch_size=21, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# make a prediction



yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# print('test_X top 10  is ')
# print(test_X[:10, :])
# 
# print('yhat top 10  is ')
# print(yhat[:10, :])

# invert scaling for forecast
# print('before concatenate  shape is ')
# print(yhat.shape, test_X.shape)
# 
# inv_yhat = concatenate((yhat, test_X), axis=1)
# 
# print('scaler.inverse_transform shape is ')
# print(inv_yhat.shape)
# 
# 
# inv_yhat = scaler.inverse_transform(inv_yhat)
# 
# 
# print('predict  result is ')
# print(inv_yhat[:, :30])
# 
# # inv_y = concatenate((test_y, test_X), axis=1)
# # inv_y = scaler.inverse_transform(inv_y)
#  
# print('原始数据 result is ')
# print(oriTestData)
#  
#  
#  
#  
#  
# pyplot.plot(inv_yhat[-1, :30], label='result')
# pyplot.plot(oriTestData, label='ori')
# pyplot.legend()
# pyplot.show()







inv_yhat = concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
print('预测数据 TOP 30 is ')
print(inv_yhat[:size])
 
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
 
print('原始数据 TOP 30 is ')
print(inv_y[0:size])
 
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
 
 
pyplot.plot(inv_yhat[0:size], label='inv_yhat')
pyplot.plot(inv_y[0:size], label='inv_y')
# pyplot.plot(oriTestData, label='inv_y')

pyplot.legend()
pyplot.show()

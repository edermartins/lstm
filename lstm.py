import math

from LstmWrapper import LstmWrapper

'''
Setting the variables
'''
# Dataset
dataset_path = 'data/big_tech_stock_prices.csv'
# Filtering the dataset with multiple companies stocks
filter_stocks = {'filed_name': 'stock_symbol', 'field_text':'GOOGL'}
# Number os splits to generate the train and test use by TimeSeriesSplit
n_splits = 20
# Number of hidden layers
hidden_layers = 4
# Number of neurons for each layer
units = 300
# Number of epochs
epochs = 300
# Verbose 0 = none, 1 more detail, 2 a lot of details and go on til 4
verbose = 0
# Selecting the Features (x)
features = ['open', 'high', 'low', 'volume']
# Select predict values (y)
y_value = 'adj_close'

'''
Starting code
'''
# Creating the wrapper
lstm_wrapper = LstmWrapper()

# Creating and setting the LSTM model
lstm_wrapper.create_setting_model(dataset_path=dataset_path, filter_stocks=filter_stocks, features=features, y_value=y_value, n_splits=n_splits, hidden_layers=hidden_layers, units=units)

# Training the model
lstm_wrapper.fit(epochs=epochs)

# Evaluating the model
score = lstm_wrapper.evaluate(verbose=verbose)
trainScore = score['train']
testScore = score['test']
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

'''
Some testes that I did

10 units 10 epochs
Train Score: 1086.63 MSE (32.96 RMSE)
Test Score: 3021.13 MSE (54.96 RMSE)

10 units 300 epochs
Train Score: 1120.89 MSE (33.48 RMSE)
Test Score: 3452.70 MSE (58.76 RMSE)

100 units 10 epochs
Train Score: 1103.54 MSE (33.22 RMSE)
Test Score: 2968.12 MSE (54.48 RMSE)

100 units 300 epochs
Train Score: 1118.98 MSE (33.45 RMSE)
Test Score: 3381.42 MSE (58.15 RMSE)

100 units 300 epochs
Train Score: 1206.03 MSE (34.73 RMSE)
Test Score: 1298.39 MSE (36.03 RMSE)
'''

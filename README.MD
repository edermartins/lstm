# Wrapper LSTM (Long Short-Term Memory)

This is a simple wrapper to make easy to use LSTM for study or real use in a real world

# Requiments

## Python

- Python 3.11 + pip

## Miniconda
Before any installation I had installed Miniconda to use the Venv: https://docs.anaconda.com/miniconda/miniconda-install/

After that I create a virtual env and installed everything inside this `venv`: `conda create --name tf-wsl python=3.11`

Activate this virtual env and install all you needs: `conda activate tf-wsl`

## Main Libraries

- Keras 3.5.0: `pip install keras`
- SKLearn: `pip install sklearn`
- Pandas: `pip install pandas`
- Numpy: `pip install numpy`
- More libraries: `pip install matplotlib seaborn pandas_datareader jupyter jupyterlab`

## NVIDIA CUDA

I recommend to use NVIDIA CUDA, otherwise will take a lot of time to train in a normal AMD or Intel CPU: https://developer.nvidia.com/cuda-downloads

# How to use

## Running the `lstm.py` file

Just run the `lstm.py` file using you local python

`python lstm.py`

If you a `conda` virtual environment:

`conda run -n <your virtual environment> --no-capture-output python ./lstm.py `

or activate the `venv` and run the `lstm.py` file (in my case, the virtual environment is tf-wsl):
```bash
# conda activate tf-wsl
(tf-wsl) # python lstm.py

```


## Use the wrapper inside you application

```python
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
lstmWrapper = LstmWrapper()

# Creating and setting the LSTM model
lstmWrapper.create_setting_model(dataset_path=dataset_path, filter_stocks=filter_stocks, features=features, y_value=y_value, n_splits=n_splits, hidden_layers=hidden_layers, units=units)

# Training the model
lstmWrapper.fit(epochs=epochs)

# Evaluating the model
score = lstmWrapper.evaluate(verbose=verbose)
trainScore = score['train']
testScore = score['test']
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
```

## Start API services: `lstm-sercice.py`

This service was made with Flask. So you need to install it: `pip install flask`
Just run the script and it will be waiting for API REST calls:
` python lstm-sercice.py`

Or use Conda command:

`conda run -n tf-wsl --no-capture-output python -m flask run`

### Endpoints available

#### POST model 
Create the model using the body params.

Request:
```json
{
    "filter_stocks": {"filed_name": "stock_symbol", "field_text":"GOOGL"},
    "features": ["open", "high", "low", "volume"],
    "y_value": "adj_close", 
    "n_splits":20, 
    "hidden_layers":4, 
    "units":100
}
```
Responses:

- 201
```json
{
    "message": "Model LSTM created with success"
}
```
- Otherwise, is an error

#### POST model/fit 
Train the model created in `POST /model`. As the training can get a lot of time, this endpoint is async. If you get code 201, you need to try `/model/evaluate` till it gets success. 

Request:
```json
{
    "epochs": 10
}
```
Responses:
- 201 This is a Async call (if you try again will get an error)
```json
{
    "message": "Model LSTM training in progress"
}
```
- 402
```json
{
    "message": "message"
}
```
Messages:
- `Model wasn't created`
- `Model still training in progress`

#### POST model/evaluate 
This function evaluate your model when ready as the endpoint `/model/fit` is asynx. After training the model and calling `/model/fit`, you need to try this method till you get success.

Request:
```json
{
    "verbose": 0
}
```

Responses:
- 201
```json
{
    "test_score": {
        "mse": 4157.3818359375,
        "rmse": 64.47776233661882
    },
    "train_score": {
        "mse": 1187.6351318359375,
        "rmse": 34.46208252320132
    }
}
```
- 422
```json
{
    "message": "message": "Model wasn't trained"
}
```

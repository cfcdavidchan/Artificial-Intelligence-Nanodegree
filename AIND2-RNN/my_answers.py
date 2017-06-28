import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series) - window_size):
        X.append(series[i : i+window_size])
        y.append(series[i+window_size])
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    np.random.seed(0)
    model = Sequential()
    model.add(LSTM(5, input_shape= X_train.shape[1:]))
    model.add(Dense(len(X_train[0][0])))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)



### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique_characters = []
    list_text = text.split()
    text_dict = dict(Counter(list_text))
    for k,v in text_dict.items():
        if v == 1:
            unique_characters.append(k)
    symbols = '${}()[].,:;@"!%?+-*/&|<>=~1234567890'
    for s in symbols:
        print('Now removing %s'%s, end='\r')
        sys.stdout.flush()
        text = text.replace(s,' ')
        time.sleep(0.1)
    text = text.replace('  ',' ')
    # remove as many non-english characters and character sequences as you can 


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    i = 0
    while i + window_size < len(text):
        string = text[i:i + window_size]
        inputs.append(string)
        outputs.append(text[i + window_size])
        i += step_size
        
    return inputs,outputs

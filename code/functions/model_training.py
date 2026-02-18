import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Conv1D, BatchNormalization, MaxPooling1D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class LSTMStockModel():
    def __init__(self) -> None:
        pass

    def loaddata(self, df, cols, training_start_date, training_cols):
        '''
        original contained components: 'Date', 'Close', 'Volume', 'compound'
        # df1 is the used data columns, training data is the values for training in the array format
        '''
        data = df[cols].set_index('Date')
        data = data.loc[data.index >= training_start_date]
        arrdata = data[training_cols].values
        return data, arrdata
    
    def training_length(self, data, historical_end_date):
        #historical_end_date, forcast_start_date = '2023-03-31', '2023-04-01'
        train_len = len(data.loc[data.index <= historical_end_date].reset_index(drop = True))  
        return train_len
    
    def normalization(self, data):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

   
    def train_test_split(self, scaled_data, arrdata, len_training, lookback=20):
        #Training set
        train_data = scaled_data[:len_training,:]  #len_training lengths, all cols
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []
        for i in range(lookback, len(train_data)):
            x_train.append(train_data[i-lookback:i,:])
            y_train.append(train_data[i, 0])


        # Testing set
        test_data = scaled_data[len_training-lookback: , :]
        x_test = []
        y_test = arrdata[len_training:, 0]
        for i in range(lookback, len(test_data)):
            x_test.append(test_data[i-lookback:i,:])


        # Convert the data sets to numpy arrays 
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)
        return x_train, y_train, x_test, y_test
    

    def model_architecture_1dim(self, x_train):
        # Build LSTM model
        model = Sequential()
        #Adding 3 LSTM layers and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1), activation='tanh'))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True, activation='tanh'))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, activation='tanh'))
        model.add(Dropout(0.2))
        # Adding the output layer
        model.add(Dense(units = 1))
        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.summary()
        return model
    

    def model_architecture_2dim(self, x_train):
        # Build LSTM model
        model = Sequential()
        #Adding 3 LSTM layers and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2]), activation='tanh'))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True, activation='tanh'))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, activation='tanh'))
        model.add(Dropout(0.2))
        # Adding the output layer
        model.add(Dense(units = 1))
        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.summary()
        return model
    
    def cnn_lstm_architecture_1dim(self, x_train):
        """Optimized CNN-LSTM for 1D (close price only) input."""
        from keras.optimizers import Adam
        model = Sequential()
        # Conv1D - reduced filters, larger kernel for better pattern capture
        model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
                         input_shape=(x_train.shape[1], 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        # LSTM layers - increased units for better representation
        model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))
        # Lower learning rate for stability
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model.summary()
        return model

    def cnn_lstm_architecture_2dim(self, x_train):
        """CNN-LSTM for 2D (close + sentiment) - same as 1D but with 2 input features."""
        from keras.optimizers import Adam
        model = Sequential()
        # Same architecture as 1D but accepts 2 features
        model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
                         input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        # Same LSTM config as 1D
        model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model.summary()
        return model

    def model_fitting(self, model, x_train, y_train, model_savepath):
        # Add early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                          callbacks=[early_stop], verbose=1)
        model.save(model_savepath)
        print('model is saved!')
        return history
    
    def prediction(self, model, x_test, y_test, scaler, df, train_len):
        # Get the model's predicted price values
        predictions = model.predict(x_test)

        # Reshape predictions to remove the extra dimension
        predictions = predictions.flatten()

        # Inverse transform the predictions to get them back to the original scale
        n_features = scaler.n_features_in_
        if n_features == 1:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))[:, 0]
        else:
            pad = np.zeros((len(predictions), n_features - 1))
            predictions = scaler.inverse_transform(np.column_stack((predictions, pad)))[:, 0]

        # Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

        # extract train and valid data
        df.index = pd.to_datetime(df.index)
        train = df[:train_len].copy()
        valid = df[train_len:].copy()
        valid['Predictions'] = predictions
        
        return predictions, train, valid, rmse 

    
    def training_loss_viz(self, history):
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(history.history['loss'], label='loss', color='#9f1f31')
        ax.plot(history.history['val_loss'], label='val_loss', color='#03608c')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        plt.legend()
        plt.savefig('fig/model training/5.1 trainingloss1.png')
        plt.close(fig)
        return fig, ax
    
    def prediction_viz(self, train, valid, predictions, savepath):
        plt.figure(figsize=(16, 8))

        plt.plot(train.index, train['Close'], label='Train', color='#03608c')
        plt.plot(valid.index, valid['Close'], label='Validation', color='#eaaa60')
        plt.plot(valid.index, predictions, label='Predictions', color='#9f1f31')

        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.legend(loc='upper left')

        plt.title('Stock Prediction Forecasting')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        plt.savefig(savepath)
        plt.close()





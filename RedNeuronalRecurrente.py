import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout
import config
class RNN():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Input(shape=(config.time_step, 56)))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dropout(0.2))

        # Añadir capas densas
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    
    def train(self,X_train,y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train, batch_size=config.batch_size, epochs=config.epochs)


    def prediccion(self,X_test,y_test,y_no_scaled,evalua=True):
        X_test=np.array(X_test)
        y_test=np.array(y_test)
        predictions = self.model.predict(X_test)
        scaler = MinMaxScaler()
        y_no_scaled=np.array(y_no_scaled).reshape(-1, 1)
        scaler.fit(y_no_scaled)
        predictions=predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        loss=None
        if y_test is not None and evalua==True:
            loss = self.model.evaluate(X_test, y_test, verbose=0)
        return predictions,loss
    


    @staticmethod
    def process_data(features):
        # Normaliza los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features)
        return scaled_data
    

    
    @staticmethod
    def train_test_split(dataset,no_scaled_data,porciento_train):
        dataX, dataY,y_no_scaled = [], [], []
        for i in range(len(dataset)-config.time_step-config.predict_step):
            a = dataset[i:(i+config.time_step), :]
            dataX.append(a)
            dataY.append(dataset[i + config.time_step + config.predict_step - 1, 0])  # Precio de cierre de la última vela en la ventana de predicción
            y_no_scaled.append(no_scaled_data.iloc[i + config.time_step + config.predict_step - 1, 0])
        # Divide los datos en entrenamiento y prueba
        train_size = int(len(dataX) * porciento_train)
        X_train, X_test = dataX[:train_size], dataX[train_size:]
        y_train, y_test = dataY[:train_size], dataY[train_size:]
        y_no_scaled=y_no_scaled[train_size:]
        return X_train,X_test,y_train,y_test,y_no_scaled
    

    @staticmethod
    def get_test_data(dataset):
        dataX = []
        for i in range(len(dataset)-config.time_step-config.predict_step):
            a = dataset[i:(i+config.time_step), :]
            dataX.append(a)
        return dataX



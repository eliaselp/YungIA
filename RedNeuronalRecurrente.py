import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout
import config

import pandas_ta as ta
import psutil

def get_data_set():
    ohlcv_df = pd.read_csv("BTC_USDT_ohlcv.csv")


    # Convertir las columnas de precios y volumen a numérico
    ohlcv_df['close'] = pd.to_numeric(ohlcv_df['close'])
    ohlcv_df['high'] = pd.to_numeric(ohlcv_df['high'])
    ohlcv_df['low'] = pd.to_numeric(ohlcv_df['low'])
    ohlcv_df['open'] = pd.to_numeric(ohlcv_df['open'])
    ohlcv_df['volume'] = pd.to_numeric(ohlcv_df['volume'])
    
    ohlcv_df['RSI'] = ta.rsi(ohlcv_df['close'],length=15)

    new_columns = pd.DataFrame()
    #EMA
    for i in range(5,101,5):
        new_columns[f'EMA-{i}'] = ta.ema(ohlcv_df['close'], length=i)
    ohlcv_df = pd.concat([ohlcv_df, new_columns], axis=1)
    
    # ATR
    ohlcv_df['ATR'] = ta.atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
    
    # Eliminar las primeras filas para evitar NaNs
    ohlcv_df = ohlcv_df.dropna()
    ohlcv_df = ohlcv_df.reset_index(drop=True)  # Reset index after dropping rows
    ohlcv_df = ohlcv_df.drop('timestamp', axis=1)

    print(ohlcv_df)

    return ohlcv_df





class RNN():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Input(shape=(config.time_step, 27)))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Añadir capas densas
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        #INICIANDO PRE ENTRENAMIENTO CON DATOS HISTORICOS    
        print("obteniendo datos de csv")
        data = get_data_set()    
        
        print("Escalando los datos")
        data_scaled = RNN.process_data(data)
        
        print("Separando los datos en entrenamiento y prueba")
        X_train,X_test,y_train,y_test,y_no_scaled=RNN.train_test_split(data_scaled,data,porciento_train=0.8)

        print("Entrenando modelo")
        self.train(X_train=X_train,y_train=y_train)


    def train(self,X_train,y_train):
        # Obtener la memoria RAM disponible
        mem = psutil.virtual_memory()
        available_memory = mem.available / 2  # Usar solo el 50% de la memoria disponible

        # Calcular el tamaño de la sección basado en la memoria disponible
        section_size = int(available_memory // (X_train[0].nbytes + y_train[0].nbytes))

        num_sections = len(X_train) // section_size + (1 if len(X_train) % section_size != 0 else 0)
        
        for i in range(num_sections):
            print(f"trining: {i}/{num_sections}")
            start_idx = i * section_size
            end_idx = min((i + 1) * section_size, len(X_train))
            
            X_section = np.array(X_train[start_idx:end_idx], dtype=np.float64)
            y_section = np.array(y_train[start_idx:end_idx], dtype=np.float64)
            self.model.fit(X_section, y_section, batch_size=config.batch_size, epochs=config.epochs)


        #X_train = np.array(X_train, dtype=np.float32)
        #y_train = np.array(y_train, dtype=np.float32)
        #self.model.fit(X_train, y_train, batch_size=config.batch_size, epochs=config.epochs)




    #dudas en el escalado inverso
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



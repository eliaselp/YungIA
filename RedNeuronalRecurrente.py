import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split as tts

import dill
import config
import os
import pandas_ta as ta
import psutil

def get_data_set():
    ohlcv_df = pd.read_csv("btc_1h.csv")

    # Convertir las columnas de precios y volumen a numérico
    ohlcv_df['close'] = pd.to_numeric(ohlcv_df['close'])
    ohlcv_df['high'] = pd.to_numeric(ohlcv_df['high'])
    ohlcv_df['low'] = pd.to_numeric(ohlcv_df['low'])
    ohlcv_df['open'] = pd.to_numeric(ohlcv_df['open'])
    ohlcv_df['volume'] = pd.to_numeric(ohlcv_df['volume'])
    
    #ohlcv_df['RSI'] = ta.rsi(ohlcv_df['close'],length=15)

    #new_columns = pd.DataFrame()
    #EMA
    #for i in range(5,101,20):
    #    new_columns[f'EMA-{i}'] = ta.ema(ohlcv_df['close'], length=i)
    #ohlcv_df = pd.concat([ohlcv_df, new_columns], axis=1)
    
    # ATR
    #ohlcv_df['ATR'] = ta.atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
    
    # Eliminar las primeras filas para evitar NaNs
    ohlcv_df = ohlcv_df.dropna()
    ohlcv_df = ohlcv_df.reset_index(drop=True)  # Reset index after dropping rows
    ohlcv_df = ohlcv_df.drop('datetime', axis=1)

    print(ohlcv_df)
    

    return ohlcv_df


def build_model():
    regressor = Sequential()
    regressor.add(Input(shape=(config.time_step, 5)))
    regressor.add(LSTM(units = 50,return_sequences=True,))
    regressor.add(Dropout(rate = 0.2))

    regressor.add(LSTM(units = 50,return_sequences=True,))
    regressor.add(Dropout(rate = 0.2))

    regressor.add(LSTM(units = 50, return_sequences=True,))
    regressor.add(Dropout(rate = 0.2))


    regressor.add(LSTM(units = 50,return_sequences=False,))
    regressor.add(Dropout(rate = 0.2))
    #FINAL - OUTPUT LAYER
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = "adam", loss  = "mean_squared_error")
    return regressor
    
    



class RNN():
    def __init__(self):
        self.model = None
        #INICIANDO PRE ENTRENAMIENTO CON DATOS HISTORICOS    
        self.load_state_model()
    
        if self.model is None and config.entrenar_dataset:
            self.model = build_model()
            print("obteniendo datos de csv")
            data = get_data_set()    
            
            print("Escalando los datos")
            data_scaled = RNN.process_data(data)
            
            print("Separando los datos en entrenamiento y prueba")
            X_train,y_train,y_no_scaled=RNN.train_test_split(data_scaled,data,porciento_train=None)
                        
            print("Entrenando modelo")
            self.pre_train(X_train=X_train,y_train=y_train)
        else:
            if self.model is None:
                self.model = build_model()
            #print("YA EL MODELO EXISTE")
            #predictions,loss=self.prediccion(X_test=X_test,y_test=y_test,y_no_scaled=y_no_scaled)
            #print(f"Indice de error: {loss}")
            #input("\n[#] Precione enter para continuar")
            #AQUI VOY A HACER LAS PRUEBAS DEL MODELO CON MATPLOTLIB
            pass
            



    def pre_train(self, X_train, y_train):
        #OBTENER EL TAMAÑO DE LA RAM DISPONIBLE Y CALCULAR TAMAÑOS DE BLOQUES
        mem = psutil.virtual_memory()
        available_memory = mem.available * 0.5
        section_size = int(available_memory // (X_train[0].nbytes + y_train[0].nbytes))
        #section_size = 1000
        
        num_sections = len(X_train) // section_size + (1 if len(X_train) % section_size != 0 else 0)
        
        for i in range(num_sections):
            print(f"[##]Entrenamiento: {i+1}/{num_sections}")
            start_idx = i * section_size
            end_idx = min((i + 1) * section_size, len(X_train)-1)
            
            X_section = np.array(X_train[start_idx:end_idx], dtype=np.float64)
            y_section = np.array(y_train[start_idx:end_idx], dtype=np.float64)
                
            self.model.fit(X_section, y_section, batch_size=config.batch_size, epochs=config.epochs)
            
        self.save_state_model()


     #LISTO
    def save_state_model(self):
        with open('00_modelo.pkl', 'wb') as file:
            dill.dump(self.model, file)
        
    #LISTO
    def load_state_model(self):
        if os.path.exists('00_modelo.pkl'):
            with open('00_modelo.pkl', 'rb') as file:
                self.model=dill.load(file)




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
            
        self.save_state_model()




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
        # Divide los datos en entrenamiento y prueba, preparando serie temporal
        dataX, dataY, y_no_scaled = [], [], []
        for i in range(len(dataset) - config.time_step - config.predict_step):
            a = dataset[i:(i + config.time_step), :]
            dataX.append(a)
            dataY.append(dataset[i + config.time_step + config.predict_step - 1, 0])  # Precio de cierre de la última vela en la ventana de predicción
            y_no_scaled.append(no_scaled_data.iloc[i + config.time_step + config.predict_step - 1, 0])
        # Utiliza train_test_split de sklearn
        if not porciento_train is None:
            X_train, X_test, y_train, y_test = tts(dataX, dataY, train_size=porciento_train, random_state=42)
        
            y_no_scaled_test = y_no_scaled[len(y_train):]
            return X_train,X_test,y_train,y_test,y_no_scaled_test
        return dataX,dataY,y_no_scaled
    

    @staticmethod
    def get_test_data(dataset):
        dataX = []
        for i in range(len(dataset)-config.time_step-config.predict_step):
            a = dataset[i:(i+config.time_step), :]
            dataX.append(a)
        return dataX



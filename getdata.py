import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Configuración del exchange
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '5m'

# Función para obtener datos históricos
def fetch_ohlcv(symbol, timeframe, since):
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # incrementar tiempo para el siguiente conjunto de datos
    return all_ohlcv

# Tiempo desde 2016
print("Tiempo desde 2016")
since = exchange.parse8601('2016-01-01T00:00:00Z')

# Obtener datos
print("Obtener datos")
ohlcv = fetch_ohlcv(symbol, timeframe, since)

# Convertir a DataFrame
print("Convertir a DataFrame")
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Exportar a CSV
print("Exportar a CSV")
df.to_csv(f'{symbol.replace("/", "")}_ohlcv.csv', index=False)

print(f"Datos OHCLV de {symbol} exportados a {symbol.replace('/', '')}_ohlcv.csv")

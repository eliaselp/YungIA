

####CONFIGURACION####

############################
#### solicitud de datos ####
############################
access_id = "CA67B0440BB547499F0BF8A632741AF0"  # Replace with your access id
secret_key = "5BBFB51795FDCFB32777F34F113720EC9C2598905DBD679F"  # Replace with your secret key
simbol="BTCUSDT"
size=1000
temporalidad="5min"
'''
 ["1min", "3min", "5min", "15min", "30min", "1hour", "2hour", "4hour", "6hour", "12hour" , "1day", "3day", "1week"]
'''






ENVIO_MAIL=True
email="liranzaelias@gmail.com"
Operar=False
entrenar_dataset = True
incluir_precio_actual=False
notificar_monitoreo = False
tiempo_espera=10 #segundos



#### API ELIAS IA ####
url_base = "https://tradingbot.ddns.net"
#url_base = "http://localhost:8000"
uid = '2d255a0b'
api_private_key_sign = '''-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgsSUgOg7hw9fqtO81
3J87xQZloSw7/InDzSEnElgowvOhRANCAAQHd7c8v1A0D12MKSY0J6GtnjVMTUnI
NnEfgi15KLCEpGfLCS/FrVDae1JpCda+5fyim3b06O7Q4HYJpIUZlMvF
-----END PRIVATE KEY-----
'''
api_public_key_auth = '''-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEB3e3PL9QNA9djCkmNCehrZ41TE1J
yDZxH4IteSiwhKRnywkvxa1Q2ntSaQnWvuX8opt29Oju0OB2CaSFGZTLxQ==
-----END PUBLIC KEY-----
'''











#CONFIG RED NEURONAL RECURRENTE
batch_size=1
epochs=1

time_step=200
predict_step=2

reset_model = 0

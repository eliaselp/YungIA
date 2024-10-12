

####CONFIGURACION####

############################
#### solicitud de datos ####
############################
access_id = "CA67B0440BB547499F0BF8A632741AF0"  # Replace with your access id
secret_key = "5BBFB51795FDCFB32777F34F113720EC9C2598905DBD679F"  # Replace with your secret key
simbol="BTCUSDT"
size=1000
temporalidad="15min"
'''
 ["1min", "3min", "5min", "15min", "30min", "1hour", "2hour", "4hour", "6hour", "12hour" , "1day", "3day", "1week"]
'''



ENVIO_MAIL=True
email="liranzaelias@gmail.com"
Operar=False
step_training = 30
entrenar_dataset = True
incluir_precio_actual=False
tiempo_espera=10 #segundos



#### API ELIAS IA ####
#url_base = "https://yungia.ddns.net"
url_base = "http://localhost:8000"
uid = '8b5cb1ae'
api_private_key_sign = '''-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgZVlamQ/EJgY8fKMv
eD/QWGWTpr1KEADt3tgKqJf0SF6hRANCAASQ5TIBYKu1VCeUSx7qyzPZ2u4JH95H
L4lKMQUT2HJgxd0Gq/RsuP0NDnR1qq+8Bk4qL5KA2JUX9iLzy3xHKGGc
-----END PRIVATE KEY-----
'''
api_public_key_auth = '''-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEkOUyAWCrtVQnlEse6ssz2druCR/e
Ry+JSjEFE9hyYMXdBqv0bLj9DQ50daqvvAZOKi+SgNiVF/Yi88t8RyhhnA==
-----END PUBLIC KEY-----
'''











#CONFIG RED NEURONAL RECURRENTE
batch_size=1
epochs=1

time_step=100
predict_step=3

reset_model = 0

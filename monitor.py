import json
import base64
import requests
import config

import ecc



    
    
    

# Método para cifrar en base64
def cifrar_base64(texto):
    texto_bytes = texto.encode('utf-8')
    base64_bytes = base64.b64encode(texto_bytes)
    base64_texto = base64_bytes.decode('utf-8')
    return base64_texto

# Método para descifrar desde base64
def descifrar_base64(base64_texto):
    base64_bytes = base64_texto.encode('utf-8')
    texto_bytes = base64.b64decode(base64_bytes)
    texto = texto_bytes.decode('utf-8')
    return texto


def dict_a_base64(diccionario):
    # Convertir el diccionario a una cadena JSON
    json_str = json.dumps(diccionario)
    # Codificar la cadena JSON en base64
    base64_str = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    return base64_str


####################################################################################################################################
def handcheck(public_key_temp_api):
    #crear canal de comunicacion seguro
    if public_key_temp_api is None:
        handcheck = {
            'uid':config.uid,
            'key':config.api_public_key_auth
        }
        handcheck = dict_a_base64(handcheck)
        url = f'{config.url_base}/setup_chanel/{handcheck}/'
        public_key_channel = ""
        response = requests.get(url)
        if response.status_code == 200:
            # Verificar el contenido de la respuesta
            if response.text:
                try:
                    json_data = response.json()
                    public_key_channel = descifrar_base64(json_data.get('public_key'))
                except ValueError as e:
                    print("Error al decodificar JSON:", e)
            else:
                print("La respuesta está vacía")
        else:
            print(f"Error en la solicitud: {response.status_code}")

        print("Public Key Channel: ",public_key_channel,end="\n")
        public_key_temp_api = public_key_channel
        return public_key_temp_api
    return None



def post_action(valor,numero_analisis,public_key_temp_api):
    public_key_channel = None
    try:
        public_key_channel = handcheck(public_key_temp_api)
        data = {
            'valor': valor,
            'numero_analisis': numero_analisis,
        }
        cyphr = dict_a_base64(data)
        cyphr = ecc.encrypt_message(public_key_channel, cyphr)
        cyphr = cifrar_base64(cyphr)

        data = {
            "data":cyphr,
            "uid":config.uid,
            "sign":ecc.sign_string(cyphr, config.api_private_key_sign)
        }

        data = dict_a_base64(data)
        url = f"{config.url_base}/update/{data}/"
        response = requests.get(url)
        # Verificar el código de estado de la respuesta

        if response.status_code == 200:
            # Verificar el contenido de la respuesta
            if response.text:
                try:
                    json_data = response.json()
                    print(json_data)
                except ValueError as e:
                    print("Error al decodificar JSON:", e)
            else:
                print("La respuesta está vacía")
        else:
            print(f"Error en la solicitud: {response.status_code}")
        
    except Exception as e:
        print(e)
    return public_key_channel



#####################################################################################################################################################################
def update_text_code(mensaje,public_key_temp_api):
    public_key_channel = None
    try:
        public_key_channel = handcheck(public_key_temp_api)
        # Ejemplo de uso
        data = {
            'mensaje': mensaje,
        }
        cyphr = dict_a_base64(data)
        cyphr = ecc.encrypt_message(public_key_channel, cyphr)
        cyphr = cifrar_base64(cyphr)

        data = {
            "data":cyphr,
            "uid":config.uid,
            "sign":ecc.sign_string(cyphr, config.api_private_key_sign)
        }

        data = dict_a_base64(data)
        url = f"{config.url_base}/update_text_code/{data}/"
        response = requests.get(url)
        # Verificar el código de estado de la respuesta

        if response.status_code == 200:
            # Verificar el contenido de la respuesta
            if response.text:
                try:
                    json_data = response.json()
                    print(json_data)
                except ValueError as e:
                    print("Error al decodificar JSON:", e)
            else:
                print("La respuesta está vacía")
        else:
            print(f"Error en la solicitud: {response.status_code}")
    except Exception as e:
        print(e)
    return public_key_channel


def update_test_predictions(prediction,current_price,predict_step,analisis,public_key_temp_api):
    public_key_channel = None
    try:
        public_key_channel = handcheck(public_key_temp_api)
        # Ejemplo de uso
        data = {
            'analisis':str(analisis),
            'prediction':str(prediction),
            'current_price':str(current_price),
            'predict_step':str(predict_step),
        }
        cyphr = dict_a_base64(data)
        cyphr = ecc.encrypt_message(public_key_channel, cyphr)
        cyphr = cifrar_base64(cyphr)

        data = {
            "data":cyphr,
            "uid":config.uid,
            "sign":ecc.sign_string(cyphr, config.api_private_key_sign)
        }
        data = dict_a_base64(data)
        url = f"{config.url_base}/update_test_predictions/{data}/"
        response = requests.get(url)
        if response.status_code == 200:
            # Verificar el contenido de la respuesta
            if response.text:
                try:
                    json_data = response.json()
                    print(json_data)
                except ValueError as e:
                    print("Error al decodificar JSON:", e)
            else:
                print("La respuesta está vacía")
        else:
            print(f"Error en la solicitud: {response.status_code}")
    except Exception as e:
        print(e)        
    return public_key_channel


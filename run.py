import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import yung_Coinex_LocalMaxMin as yung

yung.run_bot()

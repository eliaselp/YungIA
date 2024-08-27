import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import yung_Coinex_LocalMaxMin as yung

yung.run_bot()

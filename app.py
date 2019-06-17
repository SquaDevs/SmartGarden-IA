from flask import Flask
from flask import request
from flask_cors import CORS
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from keras.models import load_model
from pandas.io.json import json_normalize
from keras import backend as K

app = Flask(__name__)
# CORS(app)

@app.route('/api/analisar', methods = ['POST'])
def analisarPendente():
    print(request.data)
    K.clear_session()

    data = json.loads(request.data)

    model = load_model('SmartGardenModelBinary.h5')

    dfTest = json_normalize(data)

    le_insumo = LabelEncoder()
    le_mes = OrdinalEncoder(categories=[['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']])
    le_aparencia = OrdinalEncoder(categories=[['murcha', 'amarelada',  'normal']])
    le_temp = OrdinalEncoder(categories=[['baixo', 'medio', 'alto']])
    le_umi = OrdinalEncoder(categories=[['baixo', 'medio', 'alto']])
    le_risco = OrdinalEncoder(categories=[['baixo', 'medio', 'alto']])
    
    le_insumo.classes_ = np.load('./pickle/insumo.npy', allow_pickle=True)
    le_mes.categories_ = np.load('./pickle/mes.npy', allow_pickle=True)
    le_aparencia.categories_ = np.load('./pickle/aparencia.npy', allow_pickle=True)
    
    le_temp.categories_ = np.load('./pickle/temp.npy', allow_pickle=True)
    le_umi.categories_ = np.load('./pickle/umi.npy', allow_pickle=True)
    le_risco.categories_ = np.load('./pickle/risco.npy', allow_pickle=True)
    
    insumo = le_insumo.transform(dfTest['insumo'])
    mes = le_mes.transform(dfTest[['mes']])
    aparencia = le_aparencia.transform(dfTest[['aparencia']])
    
    dfTest['insumo'] = insumo
    dfTest['mes'] = mes
    dfTest['aparencia'] = aparencia

    predictions = model.predict(dfTest.values)

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    
    nn_preds = pd.DataFrame(predictions, columns=[0, 1, 2, 
                                                  0, 1, 2, 
                                                  0, 1, 2])

    nn_preds['encGeral'] = (nn_preds.iloc[:, 0:3] == 1).idxmax(1)
    nn_preds['encUmi'] = (nn_preds.iloc[:, 3:6] == 1).idxmax(1) 
    nn_preds['encTemp'] = (nn_preds.iloc[:, 6:9] == 1).idxmax(1)

    nn_preds['stRisco'] = le_risco.inverse_transform(nn_preds[['encGeral']])
    nn_preds['stUmi'] = le_umi.inverse_transform(nn_preds[['encUmi']])
    nn_preds['stTemp'] = le_temp.inverse_transform(nn_preds[['encTemp']])
    nn_preds = nn_preds[['stRisco', 'stUmi', 'stTemp']]

    nn_preds = nn_preds.to_json()
    nn_preds = json.loads(nn_preds)
        
    nn_preds['stRisco'] = nn_preds['stRisco']['0']
    nn_preds['stTemp'] = nn_preds['stTemp']['0']
    nn_preds['stUmi'] = nn_preds['stUmi']['0']

    return json.dumps(nn_preds)
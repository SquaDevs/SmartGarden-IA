import numpy as np
import pandas as pd
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping

data = pd.read_json('dsHortalicasV1.json')

le_insumo = LabelEncoder()

le_mes = OrdinalEncoder(categories=[['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']])
le_aparencia = OrdinalEncoder(categories=[['murcha', 'amarelada',  'normal']])

le_temp = OrdinalEncoder(categories=[['baixo', 'medio', 'alto']])
le_umi = OrdinalEncoder(categories=[['baixo', 'medio', 'alto']])
le_risco = OrdinalEncoder(categories=[['baixo', 'medio', 'alto']])

le_temp = le_temp.fit(data[['stTemp']])
le_umi = le_umi.fit(data[['stUmi']])
le_risco = le_risco.fit(data[['stRisco']])

le_mes = le_mes.fit(data[['mes']])
le_aparencia = le_aparencia.fit(data[['aparencia']])
le_insumo = le_insumo.fit(data['insumo'])

np.save('./pickle/temp.npy', le_temp.categories_)
np.save('./pickle/umi.npy', le_umi.categories_)
np.save('./pickle/risco.npy', le_risco.categories_)
np.save('./pickle/mes.npy', le_mes.categories_)
np.save('./pickle/aparencia.npy', le_aparencia.categories_)
np.save('./pickle/insumo.npy', le_insumo.classes_)

sttemp = le_temp.transform(data[['stTemp']])
stumi = le_umi.transform(data[['stUmi']])
strisco = le_risco.transform(data[['stRisco']])

mes = le_mes.transform(data[['mes']])
aparencia = le_aparencia.transform(data[['aparencia']])
insumo = le_insumo.transform(data['insumo'])

new_data = data.copy()

new_data['stTemp'] = sttemp
new_data['stUmi'] = stumi
new_data['stRisco'] = strisco
new_data['mes'] = mes
new_data['insumo'] = insumo
new_data['aparencia'] = aparencia

X_train, X_test, y_train, y_test = train_test_split(new_data.drop(columns=['stRisco', 'stUmi', 'stTemp']), new_data[['stRisco', 'stUmi', 'stTemp']], 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True)

enc = OneHotEncoder()
enc.fit(y_train)
ny_train = enc.transform(y_train).toarray()
ny_test = enc.transform(y_test).toarray()
ny_val = enc.transform(y_val).toarray()

model = Sequential()
model.add(Dense(32, activation = "relu", input_shape=(7,)))
model.add(Dense(64, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation = "relu"))
model.add(Dense(9, activation = "sigmoid"))


es = EarlyStopping(monitor='val_loss', 
                   min_delta=0.01, 
                   patience=3, 
                   verbose=0, 
                   mode='auto', 
                   baseline=None,
                   restore_best_weights=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(X_train, ny_train, callbacks=[es], 
          validation_data=(X_test, ny_test), 
          verbose=1, batch_size=None, epochs=9)

model.save('SmartGardenModelBinary.h5')

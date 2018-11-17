import keras as k
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import IPythonConsole
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import math

"""TRAINING DATA PREPROCESSING"""
#Convert training data, in the form of txt files of line-by-line SMILES strings and charges into arrays
with open('1K_SMILES_strings.txt') as my_file:
    SMILES_array = my_file.readlines()

with open('1K_SMILES_strings_charges.txt') as my_file:
    charges_array = my_file.readlines()

#Convert testing data, in the form of txt file of line-by-line SMILES strings into arrays
with open('10_SMILES_strings_test.txt') as my_file:
    test_SMILES_array = my_file.readlines()

#Convert each item of the training array of SMILES strings into molecules
mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in SMILES_array]

#Convert training molecules into training fingerprints
bi = {}
fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in mols]

#Convert training fingerprints into binary, and put all training binaries into arrays
np_fps_array = []
for fp in fps:
  arr = np.zeros((1,), dtype= int)
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_fps_array.append(arr)

"""TESTING DATA PREPROCESSING"""
#Convert each item of the testing array of SMILES strings into molecules
test_mols = [Chem.rdmolfiles.MolFromSmiles(test_SMILES_string) for test_SMILES_string in test_SMILES_array]

#Convert testing molecules into testing fingerprints
test_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(test_m, radius=2, nBits=256) for test_m in test_mols]

#Convert testing fingerprints into binary, and put all testing binaries into arrays
test_np_fps_array = []
for test_fp in test_fps:
  test_arr = np.zeros((1,), dtype= int)
  DataStructs.ConvertToNumpyArray(test_fp, test_arr)
  test_np_fps_array.append(test_arr)

"""NEURAL NETWORK"""
#The neural network model
model = Sequential([
    Dense(256, input_shape=(256,), activation= "relu"),
    Dense(128, activation= "sigmoid"),
    Dense(64, activation= "sigmoid"),
    Dense(34, activation= "sigmoid"),
    Dense(16, activation= "sigmoid"),
    BatchNormalization(axis=1),
    Dense(4, activation= "softmax")
])
model.summary()

#Compiling the model
model.compile(optimizer=Adam(lr=0.00001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Training the model
model.fit(np.array(np_fps_array), np.array(charges_array), validation_split=0.1, batch_size=10, epochs= 100, shuffle=True, verbose=1)

#Predictions with test dataset
predictions = model.predict(np.array(test_np_fps_array), batch_size=1, verbose=1)

for prediction in predictions:
    print (prediction)
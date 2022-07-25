from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import os

# disable all tensorflow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# sign language actions
sl_actions = np.array(['hallo', 'danke', 'vielglück', 'bitte', 'wo'])

# building sequential model for interference
model = Sequential()
model.add(LSTM(66, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(sl_actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# load pretrained weights
model.load_weights('../action.h5')

# convert tf model to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# safe conerted model
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("model converted successfully")
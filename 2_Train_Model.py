from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import time
import os


# global vars
DATA_PATH = os.path.join('MP_Data') 
NO_SEQ = 30
SEQ_LEN = 30

# sign language actions
sl_actions = np.array(['hallo', 'danke', 'vielglück', 'bitte', 'wo'])

# create labels and features from dataset
label_map = {label:num for num, label in enumerate(sl_actions)}
seq, labels = [], []

# load presaved training data
for action in sl_actions:
    for sequence in range(NO_SEQ):
        window = []
        for frame_num in range(SEQ_LEN):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        seq.append(window)
        labels.append(label_map[action])

# split sequence and label array in training and test data
X = np.array(seq)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# creating folder for tensorflow logs
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# building sequential model
model = Sequential()
model.add(LSTM(66, return_sequences=True, activation='relu', input_shape=(30,1662))) # 30 frames x 1662 keypoints
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu')) # fully connected layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(sl_actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# train sequential model
st = time.time()
model.fit(X_train, y_train, epochs=300, callbacks=[tb_callback])
et = time.time()

elapsed_time = et - st
print('execution time of training:', elapsed_time, 'seconds')                

# print model summary
model.summary()

model.save('action.h5')
model.load_weights('action.h5')

# evaluation using confusion matrix and accuracy with sklearn
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(f"ytrue, yhat: {(ytrue, yhat)}")
print(f"multilabel_confusion_matrix(ytrue, yhat): {multilabel_confusion_matrix(ytrue, yhat)}")
print(f"accuracy_score(ytrue, yhat): {accuracy_score(ytrue, yhat)}")

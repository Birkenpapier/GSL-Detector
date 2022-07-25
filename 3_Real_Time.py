from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import utils.drawing as draw
import numpy as np
import cv2


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

# load saved weights
model.load_weights('action.h5')

sequence = []
sl_action = ""

cap = cv2.VideoCapture(0)

# creating mediapipe holistic model 
with draw.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # infinite loop unitl camera stream is closed
    while cap.isOpened():
        _, frame = cap.read()

        # detections with mediapipe
        img, res = draw.mediapipe_detection(frame, holistic)
        draw.draw_customized_landmarks(img, res)
        
        keypoints = draw.convert_keypoints(res)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        # predict the sign language action in real time
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(f"prediction: {str(sl_actions[np.argmax(res)])}")
            sl_action = str(sl_actions[np.argmax(res)])
            
        # workaround due to opencv cannot encode utf-8 umlauts
        if sl_action == "vielglück":
            sl_action = "vielglueck"

        # write the prediction in the image and show it to the user
        cv2.rectangle(img, (0,0), (160, 40), (0, 0, 0), -1)
        cv2.putText(img, sl_action, (3,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # show screen
        cv2.imshow('Detection', img)

        # gracefully exiting app
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # release all ressources
    cap.release()
    cv2.destroyAllWindows()

# release all ressources
cap.release()
cv2.destroyAllWindows()

import utils.drawing as draw
import tensorflow as tf
import numpy as np
import cv2
import os

# disable all tensorflow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# sign language actions
sl_actions = np.array(['hallo', 'danke', 'vielglück', 'bitte', 'wo'])
seq = []

cap = cv2.VideoCapture(0)
with draw.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        # Make detections
        image, results = draw.mediapipe_detection(frame, holistic)
        # Draw landmarks
        draw.draw_customized_landmarks(image, results)
        # Prediction logic
        keypoints = draw.convert_keypoints(results)
        seq.append(keypoints)
        seq = seq[-30:]
        
        if len(seq) == 30:
            # converting input data to float32
            input_data_real = np.expand_dims(seq, axis=0)
            input_data_real = input_data_real.astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data_real)
            
            # call the interpreter
            interpreter.invoke()
            
            # get_tensor() returns a copy of the tensor data
            res = interpreter.get_tensor(output_details[0]['index'])
            print(f"prediction: {str(sl_actions[np.argmax(res)])}")

            sl_action = str(sl_actions[np.argmax(res)])
                
            # workaround due to opencv cannot encode utf-8 umlauts
            if sl_action == "vielglück":
                sl_action = "vielglueck"

            # write the prediction in the image and show it to the user
            cv2.rectangle(image, (0,0), (160, 40), (0, 0, 0), -1)
            cv2.putText(image, sl_action, (3,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # show screen
        cv2.imshow('Detection', image)

        # gracefully exiting app
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # release all ressources
    cap.release()
    cv2.destroyAllWindows()

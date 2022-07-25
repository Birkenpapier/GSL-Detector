import utils.drawing as draw
import numpy as np
import time
import cv2
import os


# global vars
DATA_PATH = os.path.join('MP_Data') 
NO_SEQ = 30
SEQ_LEN = 30

# sign language actions
sl_actions = np.array(['hallo', 'danke', 'vielglück', 'bitte', 'wo'])

# create folder for every sign language action
for a in sl_actions: 
    for seq in range(NO_SEQ):
        try: 
            os.makedirs(os.path.join(DATA_PATH, a, str(seq)))
        except:
            pass

cap = cv2.VideoCapture(0)

# creating mediapipe holistic model 
with draw.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # collect data for every sign language action
    for a in sl_actions:
        et_time = 0

        # loop through sequences
        for seq in range(NO_SEQ):
            st = time.time()

            # loop through sequence length
            for no_frame in range(SEQ_LEN):
                ret, frame = cap.read()

                # mediapie holistic detections
                img, res = draw.mediapipe_detection(frame, holistic)

                # mediapie holistic keypoints
                draw.draw_customized_landmarks(img, res)
                
                cv2.rectangle(img, (0,0), (350, 18), (0, 0, 0), -1)
                if no_frame == 0: 
                    cv2.putText(img, 'CAP START', (50,280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255, 255), 15, cv2.LINE_AA)
                    cv2.putText(img, f'sequence for --{a}-- no_seq: {seq}', (20,13), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # show screen of webcam
                    cv2.imshow('Create Dataset', img)
                    cv2.waitKey(1500)
                else: 
                    cv2.putText(img, f'sequence for --{a}-- no_seq: {seq}', (20,13), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # show screen of webcam
                    cv2.imshow('Create Dataset', img)
                
                keypoints = draw.convert_keypoints(res)
                npy_path = os.path.join(DATA_PATH, a, str(seq), str(no_frame))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            et = time.time()

            # get the execution time
            elapsed_time = et - st
            et_time += elapsed_time
            print('execution time of sequence:', elapsed_time, 'seconds')                
                    
        print(f'average et_time: {et_time / NO_SEQ} seconds')                

    # release all ressources
    cap.release()
    cv2.destroyAllWindows()

# release all ressources
cap.release()
cv2.destroyAllWindows()

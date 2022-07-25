import mediapipe as mp
import numpy as np
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(img, model):
    '''
    Returns the predicion from mediapipe model

            Parameters:
                    img (cv::Mat): opencv image matrix
                    model (mp_holistic.Holistic): mediapipe holistic model

            Returns:
                    img (cv::Mat): opencv image matrix
                    res (mp.solutions): keypoints detected by mediapipe
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert color space from BGR to RGB
    res = model.process(img) # mediapipe prediction
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # undo color space conversion

    return img, res

def draw_customized_landmarks(img, res):
    '''
    Customize the keypoint visualization from mediapipe

            Parameters:
                    img (cv::Mat): opencv image matrix
                    res (mp.solutions): keypoints detected by mediapipe
    '''
    # face
    mp_drawing.draw_landmarks(img, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,255,120), thickness=1, circle_radius=1)
                             )

    # pose
    mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,45,120), thickness=2, circle_radius=2)
                             )

    # left hand
    mp_drawing.draw_landmarks(img, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(120,20,75), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(120,45,250), thickness=2, circle_radius=2)
                             )

    # right hand  
    mp_drawing.draw_landmarks(img, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,120,65), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,65,230), thickness=2, circle_radius=2)
                             ) 

def convert_keypoints(res):
    '''
    Returns the extraced keypoints from mediapipe detection

            Parameters:
                    res (mp.solutions): keypoints detected by mediapipe

            Returns:
                    k_points (np.arr): concatenated numpy array of all keypoints
    '''
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in res.pose_landmarks.landmark]).flatten() if res.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in res.face_landmarks.landmark]).flatten() if res.face_landmarks else np.zeros(468*3)
    l_hand = np.array([[res.x, res.y, res.z] for res in res.left_hand_landmarks.landmark]).flatten() if res.left_hand_landmarks else np.zeros(21*3)
    r_hand = np.array([[res.x, res.y, res.z] for res in res.right_hand_landmarks.landmark]).flatten() if res.right_hand_landmarks else np.zeros(21*3)
    
    k_points = np.concatenate([pose, face, l_hand, r_hand])

    return k_points

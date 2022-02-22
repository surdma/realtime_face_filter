import os
import cv2
import time
import numpy as np
import mediapipe as mp 

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh ( 
    max_num_faces=10,
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

mpFaceDetection = mp.solutions.face_mesh
meshDraw = mp.solutions.drawing_utils
meshDetect = mpFaceDetection.FaceMesh(max_num_faces=2)
draw_setings = meshDraw.DrawingSpec(thickness=1, circle_radius=1)

def get_landmarks(frame):
    landmarks = []
    height, width = frame.shape[0:2]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            current = {}
            meshDraw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            for i, landmark in enumerate(face_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                relative_x = int(x * width)
                relative_y = int(y * height)
                current[i + 1] = (relative_x, relative_y)
            landmarks.append(current)
            
    return landmarks

def get_effect_cordinates(landmarks):
    effect_cordinates = {
        "eye_left": (landmarks[30], (landmarks[158][0], landmarks[145][1])),
        "eye_right": (landmarks[287], (landmarks[260][0], landmarks[381][1])),
        "shade": (landmarks[71], (landmarks[294][0], landmarks[119][1])),
        "nose": ((landmarks[51][0], landmarks[4][1]), (landmarks[281][0], landmarks[3][1])),
        "cigar": (landmarks[16], (landmarks[273][0], landmarks[195][1])),
        "mustache": ((landmarks[148][0], landmarks[3][1]), ((landmarks[148][0]+(landmarks[3][0]-landmarks[148][0])*2), landmarks[41][1])),
        "mask": (landmarks[124], (landmarks[324][0], landmarks[153][1]))
    }
    
    return effect_cordinates


def app(source):
    display = np.ones((650, 1300, 3), dtype="uint8")
    # prev_frame_time, current_frame_time, fps = 0, 0, 0
    cap = cv2.VideoCapture(source)
    
    while True:
        success, frame = cap.read()

        if success:
            current_time = time.time()
            height, width, _ = frame.shape
            image = cv2.resize(frame, (950, 650))
            
            landmarks = get_landmarks(image)
            faces = len(landmarks)

            if faces > 0:
                for l in landmarks:
                    cordinates = get_effect_cordinates(l)
                    # draw_face_effects(image, cordinates)
                display[:, 350:, :] = image


        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


app(2)
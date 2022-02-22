import os
import cv2 # pip install opencv-python
import time
import numpy as np # pip install numpy
import mediapipe as mp # pip install mediapipe

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh (
    max_num_faces=10,
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

num_icons = []
effect_icons = {}
icon_root = "../dataset"
num_icon_root = "../dataset/ui/nums"
num_icon_path = os.path.join(icon_root, "ui/nums")
default_icon_path = os.path.join(icon_root, "ui/na.png")
files = os.listdir(num_icon_root)
effects = ["eye", "shade", "nose", "cigar", "mustache", "mask"]
current_effect = None
effect_icon_counter = {
    "eye": 0,
    "shade": 0,
    "nose": 0,
    "cigar": 0,
    "mustache": 0,
    "mask": 0
}
current_effect_icons = {
    "eye": None,
    "shade": None,
    "nose": None,
    "cigar": None,
    "mustache": None,
    "mask": None
}
effect_commands = {
    ord('1'): "eye",
    ord('2'): "shade",
    ord('3'): "nose",
    ord('4'): "mustache",
    ord('5'): "cigar",
    ord('6'): "mask",
}
status_panel_effect_icon_cordinates = {
    "eye": {'y': 340, "y+h": 370, 'x': 250, "x+w": 310},
    "shade": {'y': 385, "y+h": 415, 'x': 250, "x+w": 310},
    "nose": {'y': 430, "y+h": 460, 'x': 250, "x+w": 310},
    "mustache": {'y': 475, "y+h": 505, 'x': 250, "x+w": 310},
    "cigar": {'y': 520, "y+h": 550, 'x': 250, "x+w": 310},
    "mask": {'y': 565, "y+h": 595, 'x': 250, "x+w": 310}
}
inc_dec_commands = [ord('+'), ord('-')]

for effect in effects:
    icons = os.listdir(os.path.join(icon_root, effect))
    effect_icons[effect] = icons

for file in files:
    icon = cv2.imread(os.path.join(num_icon_root, file))
    icon = cv2.resize(icon, (30, 30))
    num_icons.append(icon)


def get_landmarks(image):
    landmarks = []
    height, width = image.shape[0:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mesh_results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            current = {}
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
        "mustache": ((landmarks[148][0], landmarks[3][1]),
                     ((landmarks[148][0] + (landmarks[3][0] - landmarks[148][0]) * 2), landmarks[41][1])),
        "mask": (landmarks[124], (landmarks[324][0], landmarks[153][1]))
    }

    return effect_cordinates

def remove_image_whitespace(image, blend, x, y, threshold=225):
    for i in range(blend.shape[0]):
        for j in range(blend.shape[1]):
            for k in range(3):
                if blend[i][j][k] > threshold:
                    blend[i][j][k] = image[i + y][j + x][k]


def add_effect(image, effect, icon_path, cordinates):
    item = cv2.imread(icon_path)
    pt1, pt2 = cordinates[effect]
    x, y, x_w, y_h = pt1[0], pt1[1], pt2[0], pt2[1]
    cropped = image[y:y_h, x:x_w, :]
    h, w, _ = cropped.shape
    item = cv2.resize(item, (w, h))
    blend = cv2.addWeighted(cropped, 0, item, 1.0, 0)

    return blend, x, y, x_w, y_h


def set_effect_icon(effect, step=1):
    effect_icon_counter[effect] += step

    if step > 0:
        if effect_icon_counter[effect] >= len(effect_icons[effect]):
            diff = abs(len(effect_icons[effect]) - effect_icon_counter[effect])
            effect_icon_counter[effect] = diff
    elif step < 0:
        if effect_icon_counter[effect] < -len(effect_icons[effect]):
            diff = abs(-len(effect_icons[effect]) - effect_icon_counter[effect])
            effect_icon_counter[effect] = len(effect_icons[effect]) - diff

    icon_name = effect_icons[effect][effect_icon_counter[effect]]
    icon_path = os.path.join(os.path.join(icon_root, effect), icon_name)
    current_effect_icons[effect] = icon_path


prev_frame_time = 0


def calc_fps(current_frame_time):
    global prev_frame_time
    fps = int(1 / (current_frame_time - prev_frame_time))
    prev_frame_time = current_frame_time

    return fps

def draw_status_panel_effect_icons(panel):
    for k, v in current_effect_icons.items():
        cor = status_panel_effect_icon_cordinates[k]
        if v is None:
            icon = cv2.imread(default_icon_path)
        else:
            icon = cv2.imread(current_effect_icons[k])
        icon = cv2.resize(icon, (60, 30))
        panel[cor['y']:cor["y+h"], cor['x']:cor["x+w"], :] = icon

def draw_face_effects(image, cordinates):
    for effect, icon_path in current_effect_icons.items():
        if effect == "eye":
            for effect in ["eye_left", "eye_right"]:
                if icon_path is not None:
                    blend, x, y, x_w, y_h = add_effect(image, effect, icon_path, cordinates)
                    remove_image_whitespace(image, blend, x, y)
                    image[y:y_h, x:x_w, :] = blend
        else:
            if icon_path is not None:
                blend, x, y, x_w, y_h = add_effect(image, effect, icon_path, cordinates)
                remove_image_whitespace(image, blend, x, y)
                image[y:y_h, x:x_w, :] = blend


def setup_status_panel(display, fps, num_faces=1, eye_font_col=(0, 0, 255), shade_font_col=(0, 0, 255),
                       nose_font_col=(0, 0, 255), mustache_font_col=(0, 0, 255),
                       cigar_font_col=(0, 0, 255), mask_font_col=(0, 0, 255)):
    display[340:370, 32:62, :] = num_icons[0]
    display[385:415, 32:62, :] = num_icons[1]
    display[430:460, 32:62, :] = num_icons[2]
    display[475:505, 32:62, :] = num_icons[3]
    display[520:550, 32:62, :] = num_icons[4]
    display[565:595, 32:62, :] = num_icons[5]

    cv2.circle(display, (170, 225), 80, (255, 0, 0), 2)
    cv2.putText(display, "FPS: {}".format(fps), (245, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 192), 1)
    cv2.putText(display, "LIVE", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    cv2.putText(display, "Face Effects", (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(display, "Faces", (125, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(display, "{:02}".format(num_faces), (128, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cv2.putText(display, "Eyes", (120, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_font_col, 1)
    cv2.putText(display, "Shade", (120, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, shade_font_col, 1)
    cv2.putText(display, "Nose", (120, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.6, nose_font_col, 1)
    cv2.putText(display, "Mustache", (120, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mustache_font_col, 1)
    cv2.putText(display, "Cigar", (120, 544), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cigar_font_col, 1)
    cv2.putText(display, "Mask", (120, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mask_font_col, 1)


def app(video_source):
    global current_effect
    pre_k = None

    display = np.ones((650, 1300, 3), dtype="uint8")
    prev_frame_time, current_frame_time, fps = 0, 0, 0
    source = cv2.VideoCapture(video_source)

    while True:
        ret, frame = source.read()
        if ret:
            current_time = time.time()
            fps = calc_fps(current_time)
            height, width, _ = frame.shape
            image = cv2.resize(frame, (950, 650))

            landmarks = get_landmarks(image)
            faces = len(landmarks)

            if faces > 0:
                for l in landmarks:
                    cordinates = get_effect_cordinates(l)
                    draw_face_effects(image, cordinates)
                display[:, 350:, :] = image

            status_panel = np.zeros((650, 350, 3))
            draw_status_panel_effect_icons(status_panel)
            display[:, :350, :] = status_panel

            if current_effect is None:
                setup_status_panel(display, fps, num_faces=faces)
            elif current_effect == "eye":
                setup_status_panel(display, fps, num_faces=faces, eye_font_col=(0, 255, 0))
            elif current_effect == "shade":
                setup_status_panel(display, fps, num_faces=faces, shade_font_col=(0, 255, 0))
            elif current_effect == "nose":
                setup_status_panel(display, fps, num_faces=faces, nose_font_col=(0, 255, 0))
            elif current_effect == "mustache":
                setup_status_panel(display, fps, num_faces=faces, mustache_font_col=(0, 255, 0))
            elif current_effect == "cigar":
                setup_status_panel(display, fps, num_faces=faces, cigar_font_col=(0, 255, 0))
            elif current_effect == "mask":
                setup_status_panel(display, fps, num_faces=faces, mask_font_col=(0, 255, 0))

            cv2.imshow("Live Face Effects", display)
            k = cv2.waitKey(1)

            if k in effect_commands:
                if k == pre_k:
                    current_effect_icons[current_effect] = current_effect = pre_k = None
                else:
                    current_effect, pre_k = effect_commands[k], k

            elif k in inc_dec_commands and current_effect is not None:
                if k == inc_dec_commands[0]:
                    set_effect_icon(current_effect)
                elif k == inc_dec_commands[1]:
                    set_effect_icon(current_effect, step=-1)
            elif k == 27:
                break
        else:
            break

    source.release()
    cv2.destroyAllWindows()

app(0)
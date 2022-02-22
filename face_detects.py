import cv2
import mediapipe as mp

class FaceAI():
    def __init__(self, confidence_level = 0.8):
        self.confidence_level = confidence_level
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.confidence_level)


    def detectFace(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []

        if self.results.detections:
            for id,detections in enumerate(self.results.detections):
                bboxC = detections.location_data.relative_bounding_box
                imgHeight, imgWidth, imgChannel = frame.shape
                bounding_box = int(bboxC.xmin * imgWidth), int(bboxC.ymin * imgHeight), int(bboxC.width * imgWidth), int(bboxC.height * imgHeight)
                bboxs.append([id,bounding_box, detections.score])
                cv2.rectangle(frame, bounding_box, (70, 120, 255))
                cv2.putText(frame, f'FPS: {int(detections.score[0] * 100)}%', (bounding_box[0], bounding_box[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (70, 120, 255), 2)

        return frame, bboxs


    def add_effect(image, effect, icon_path, cordinates):
        item = cv2.imread(icon_path)
        pt1, pt2 = cordinates[effect]
        x, y, x_w, y_h = pt1[0], pt1[1], pt2[0], pt2[1]
        cropped = image[y:y_h, x:x_w, :]
        h, w, _ = cropped.shape
        item = cv2.resize(item, (w, h))
        blend = cv2.addWeighted(cropped, 0, item, 1.0, 0)

        return blend, x, y, x_w, y_h


def main():
    #cap = cv2.VideoCapture("dataset/video1.mp4")
    cap = cv2.VideoCapture(2)
    detect = FaceAI()

    while True:
        success, frame = cap.read()
        frame,bboxs = detect.detectFace(frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

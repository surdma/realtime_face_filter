# RealTime Face Filter

### The mediapipe’s face detection solution is based on BlazeFace face detector that uses a very lightweight and highly accurate feature extraction network.


### Initialize the Mediapipe Face Detection Model

To use the Mediapipe’s Face Detection solution, we will first have to initialize the face detection class using the syntax mp.solutions.face_detection, and then we will have to call the function mp.solutions.face_detection.FaceDetection() with the arguments explained below:

### Read an Image

Now we will use the function cv2.imread() to read a sample image and then display, after converting it into RGB from BGR format.
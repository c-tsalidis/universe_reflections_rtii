import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognition
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import pythonosc

USE_WEBCAM = True  # If false, loads video file source

from pythonosc.dispatcher import Dispatcher
from typing import List, Any

dispatcher = Dispatcher()
count_before_average = 50
emotional_block = 0
counter = 0


def set_filter(address: str, *args: List[Any]) -> None:
    # We expect two float arguments
    if not len(args) == 2 or type(args[0]) is not float or type(args[1]) is not float:
        return

    # Check that address starts with filter
    if not address[:-1] == "/filter":  # Cut off the last character
        return

    value1 = args[0]
    value2 = args[1]
    filterno = address[-1]
    print(f"Setting filter {filterno} values: {value1}, {value2}")


dispatcher.map("/filter*", set_filter)  # Map wildcard address to set_filter function

# Set up server and client for testing
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

ip = "127.0.0.1"
port = 8108

client = SimpleUDPClient(ip, port)  # Create client

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("images/Obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
trump_image = face_recognition.load_image_file("images/Trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

modi_image = face_recognition.load_image_file("images/Modi.jpg")
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

snaylo_image = face_recognition.load_image_file("images/Snaylo.jpg")
snaylo_face_encoding = face_recognition.face_encodings(snaylo_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    trump_face_encoding,
    modi_face_encoding,
    snaylo_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Trump",
    "Modi",
    "Fearless Leader"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def take_average_emotion(total):
    average = total / count_before_average
    if average >= 0.5:
        client.send_message("/happy", 1)
    else:
        client.send_message("/sad", 1)
    print(average)
    global emotional_block
    emotional_block = 0
    global counter
    counter = 0


def face_compare(frame, process_this_frame):
    print("compare")
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    return face_names
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # cv2.rectangle(frame, (left, bottom+36), (right, bottom), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.3, (255, 255, 255), 1)
        print("text print")


# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0)  # Webcam source
else:
    cap = cv2.VideoCapture('./test/testvdo.mp4')  # Video file source

while cap.isOpened():  # True:
    ret, frame = cap.read()

    # frame = video_capture.read()[1]

    # To print the facial landmarks
    # landmrk = face_recognition.face_landmarks(frame)
    # for l in landmrk:
    #    for key,val in l.items():
    #        for (x,y) in val:
    #            cv2.circle(frame, (x, y), 1, (255,0, 0), -1)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)
    # face_locations = face_recognition.face_locations(rgb_image)
    # print (reversed(face_locations))
    face_name = "face"  # face_compare(rgb_image, process_this_frame)

    for face_coordinates, fname in zip(faces, face_name):
        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
            # client.send_message("/sad", 1)

        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
            # client.send_message("/sad", 1)
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
            # client.send_message("/happy", 1)
            emotional_block += 1

        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            # client.send_message("/happy", 1)
            emotional_block += 1
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
            # client.send_message("/happy", 1)
            emotional_block += 1
        color = color.astype(int)
        color = color.tolist()

        if fname == "Unknown":
            name = emotion_text
        else:
            name = str(fname) + " is " + str(emotion_text)

        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, name,
                  color, 0, -45, 0.5, 1)

        counter += 1
        if counter >= count_before_average:
            counter = 0
            take_average_emotion(emotional_block)



    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

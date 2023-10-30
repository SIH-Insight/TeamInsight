import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tempfile
from tensorflow import keras
from collections import deque
plt.style.use("seaborn")

IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence","Violence"]

def frames_extraction(video_bytes):

    frames_list = []

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        video_bytes = video_bytes or b''
        temp_file.write(video_bytes)  
        video_path = temp_file.name
    video_reader = cv2.VideoCapture(video_path)

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    print(skip_frames_window)

    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        normalized_frame = resized_frame / 255

        frames_list.append(normalized_frame)


    video_reader.release()

    return frames_list


def create_dataset(video_file_path):

    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):

        print(f'Extracting Data of Class: {class_name}')

        frames = frames_extraction(video_file_path)

        if len(frames) == SEQUENCE_LENGTH:

                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, video_files_paths


def predict_frames(video_bytes, SEQUENCE_LENGTH):
    MoBiLSTM_model = keras.models.load_model(r'src/garbage/garbage_classification_model.h5')

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_bytes)  
        video_path = temp_file.name

    video_reader = cv2.VideoCapture(video_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    predicted_class_name = ''

    while video_reader.isOpened():

        ok, frame = video_reader.read()

        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:

            predicted_labels_probabilities = MoBiLSTM_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            predicted_label = np.argmax(predicted_labels_probabilities)

            predicted_class_name = CLASSES_LIST[predicted_label]

        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

        #video_writer.write(frame)

    video_reader.release()
    #video_writer.release()

def show_pred_frames(pred_video_path):
    plt.figure(figsize=(20, 15))

    video_bytes = pred_video_path.read()

    # Convert the byte string to a numpy array
    video_array = np.frombuffer(video_bytes, np.uint8)

    # Use cv2.VideoCapture on the numpy array
    video_reader = cv2.VideoCapture(video_array.tobytes())
    frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    random_range = sorted(random.sample(range(SEQUENCE_LENGTH, frames_count), 12))

    for counter, random_index in enumerate(random_range, 1):
        plt.subplot(5, 4, counter)

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, random_index)

        ok, frame = video_reader.read()

        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.imshow(frame)
        plt.gca().set_aspect('equal')  # Equal aspect ratio for images
        plt.axis('off')  # Turn off axis labels and ticks

    video_reader.release()

    plt.tight_layout()
    plt.show()

def predict_video(video_bytes, SEQUENCE_LENGTH):
    MoBiLSTM_model = keras.models.load_model(r'src/garbage/garbage_classification_model.h5')
    # Use cv2.VideoCapture on the numpy array
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_bytes)  
        video_path = temp_file.name
    video_reader = cv2.VideoCapture(video_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_list = []

    predicted_class_name = ''

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH*5),1)

    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (64,64))

        normalized_frame = resized_frame / 255

        frames_list.append(normalized_frame)

    if len(frames_list) == 0:
        print("No frames to predict.")
    else:
        if(len(frames_list) < SEQUENCE_LENGTH):
            frames_list.extend([frames_list[-1]]*(SEQUENCE_LENGTH-len(frames_list)))
        
        # Convert frames_list to a NumPy array and add the batch dimension
        input_data = np.expand_dims(np.array(frames_list), axis=0)

        # Make predictions
        predicted_labels_probabilities = MoBiLSTM_model.predict(input_data)[0]

        predicted_label = np.argmax(predicted_labels_probabilities)

        predicted_class_name = CLASSES_LIST[predicted_label]

        print(f'Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        return predicted_class_name, predicted_labels_probabilities[predicted_label];


    video_reader.release()



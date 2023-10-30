import cv2
import numpy as np
import pandas as pd
import tempfile
import os

def work_management(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_bytes)
        video_path = temp_file.name

    weights_path = os.path.join('src', 'crowd', 'yolov4-tiny.weights')
    cfg_path = os.path.join('src', 'crowd', 'yolov4-tiny.cfg')

    # Load YOLOv4 model
    net = cv2.dnn.readNet(weights_path, cfg_path)
    cap = cv2.VideoCapture(video_path)

    timestamps = []
    human_counts = []
    temp_human_counts = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        human_count = 0
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.60 and class_id == 0:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    human_count += 1

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 5000.0
        timestamps.append(timestamp)
        temp_human_counts.append(human_count)
        frame_count += 1

        if frame_count == 10:
            average_human_count = sum(temp_human_counts) / len(temp_human_counts)
            human_counts.append(average_human_count)
            temp_human_counts = []
            frame_count = 0

    if temp_human_counts:
        average_human_count = sum(temp_human_counts) / len(temp_human_counts)
        human_counts.append(average_human_count)

    min_length = min(len(timestamps), len(human_counts))
    timestamps = timestamps[:min_length]
    human_counts = human_counts[:min_length]

    df = pd.DataFrame({'timestamp': timestamps, 'human_count': human_counts})

    cap.release()
    cv2.destroyAllWindows()

    return df


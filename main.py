import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
from food_facts import food_facts
import os
import uuid

def speech(text):
    print(text)
    language = "en"
    output = gTTS(text=text, lang=language, slow=False)
    output_path = f"./sounds/output_{uuid.uuid4().hex}.mp3"
    output.save(output_path)
    
    # Check if file is created
    if os.path.exists(output_path):
        print(f"Sound file created at: {output_path}")
        playsound(output_path)
    else:
        print("Sound file not found!")

# Test speech to ensure gTTS and playsound are working
speech("Test speech to verify gTTS and playsound functionality.")

# Define paths to model files
config_file = 'model/yolov3.cfg'
weights_file = 'model/yolov3.weights'
names_file = 'model/coco.names'

# Load YOLO
net = cv2.dnn.readNet(weights_file, config_file)
with open(names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            if label not in labels:
                labels.append(label)
                speech(f"Detected a {label}")

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

i = 0
new_sentence = []
for label in labels:
    if i == 0:
        new_sentence.append(f"I found a {label}, and, ")
    else:
        new_sentence.append(f"a {label},")
    i += 1

speech(" ".join(new_sentence))
speech("Here are the food facts I found for these items:")

for label in labels:
    try:
        print(f"\n\t{label.title()}")
        food_facts(label)
    except Exception as e:
        print(f"No food facts for this item: {label} - {e}")

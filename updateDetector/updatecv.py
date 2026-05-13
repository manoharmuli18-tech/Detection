from flask import Flask, render_template
from ultralytics import YOLO
import cv2
import threading
import gdown
import os

app = Flask(__name__)

# Google Drive file ID
FILE_ID = "1ENYmCfEQEExzLmjcpuuWBWquEAqYjxmr"

# Local model path
MODEL_PATH = "yolo26x.pt"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = YOLO(MODEL_PATH)

is_running = False


def run_detection():

    global is_running

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            frame,
            imgsz=320,
            conf=0.5,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow("YOLO Detection", annotated)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    is_running = False


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/start')
def start():

    global is_running

    if not is_running:
        is_running = True

        threading.Thread(
            target=run_detection,
            daemon=True
        ).start()

        return "Detection Started"

    return "Already Running"


if __name__ == '__main__':
    app.run(debug=True)

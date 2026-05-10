from flask import Flask, render_template
from ultralytics import YOLO
import cv2
import threading

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/file/d/1ENYmCfEQEExzLmjcpuuWBWquEAqYjxmr/view?usp=drivesdk"
MODEL_PATH = "yolo26x.pt"

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

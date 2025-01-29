import cv2
import numpy as np
import tensorflow as tf
from collections import deque


def live_violence_detection():
    print("Loading model ...")
    model = tf.keras.models.load_model('modelnew.h5')

    Q = deque(maxlen=128)
    cap = cv2.VideoCapture("Testing videos/V_743.mp4")  # 0 for webcam; replace with URL for an IP camera

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('output2.avi', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32") / 255.0
        frame = np.expand_dims(frame, axis=0)

        preds = model.predict(frame)[0]
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        label = (results > 0.50)[0]

        text_color = (0, 0, 255) if label else (0, 255, 0)
        text = f"Violence: {label}"

        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 3)
        cv2.imshow("Live Violence Detection", output)
        out.write(output)  # Save output frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Cleaning up...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_violence_detection()

import cv2
import numpy as np
import time
import os

# Kill any stuck camera processes (important)
os.system("pkill -f libcamera")

# TFLite import
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime")
except:
    import tensorflow.lite as tflite
    print("Using tensorflow.lite")

from picamera2 import Picamera2

# -------------------------
# 1. Load model
# -------------------------
print("🧠 Loading TFLite model...")

interpreter = tflite.Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, H, W, _ = input_details[0]['shape']

print(f"Model input: {H}x{W}")

# -------------------------
# 2. Start camera ONCE
# -------------------------
print("📸 Starting camera...")

picam2 = Picamera2()
picam2.start()

time.sleep(2)  # stabilize camera

# -------------------------
# 3. Continuous loop
# -------------------------
count = 0

while True:
    print(f"\n🔄 Cycle {count}")

    # Capture frame
    frame = picam2.capture_array()

    # Fix RGBA → RGB
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    orig_h, orig_w = frame.shape[:2]

    # -------------------------
    # Preprocess
    # -------------------------
    img = cv2.resize(frame, (W, H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # -------------------------
    # Inference
    # -------------------------
    start = time.time()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    end = time.time()

    print(f"⏱ Inference: {end - start:.2f} sec")

    output = interpreter.get_tensor(output_details[0]['index'])

    # -------------------------
    # Detection
    # -------------------------
    # Ensure proper format for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = np.ascontiguousarray(frame)

    detections = output[0]
    CONF_THRESH = 0.3

    defect_found = False

    for det in detections:
        conf = det[4]

        if conf > CONF_THRESH:
            defect_found = True

            x, y, w_box, h_box = det[0:4]

            x1 = int((x - w_box / 2) * orig_w)
            y1 = int((y - h_box / 2) * orig_h)
            x2 = int((x + w_box / 2) * orig_w)
            y2 = int((y + h_box / 2) * orig_h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(frame, f"Defect {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

    # If no defect
    if not defect_found:
        cv2.putText(frame, "No_defect",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3)

    # -------------------------
    # Save output
    # -------------------------
    filename = f"output_{count}.jpg"
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"💾 Saved: {filename}")

    count += 1

    # -------------------------
    # Wait 5 seconds
    # -------------------------
    time.sleep(5)
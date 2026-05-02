import cv2
import requests
import numpy as np
import time
from ultralytics import YOLO
from IPython.display import display, clear_output
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = 'best_float32.tflite'
SHOT_URL = "http://192.168.137.187:8080/shot.jpg" 
POST_PROCESS_WAIT = 5  # Fixed wait time after processing is finished
CONF_LEVEL = 0.4

# 1. Load Model
print("System Initializing...")
model = YOLO(MODEL_PATH, task='detect')

print(f"System Active. Waiting {POST_PROCESS_WAIT}s between each inspection.")

try:
    while True:
        try:
            # 2. Capture high-res photo (with high timeout to prevent errors)
            response = requests.get(SHOT_URL, timeout=20)
            
            if response.status_code == 200:
                img_array = np.array(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, -1)
                
                # 3. AI Inference
                results = model(frame, conf=CONF_LEVEL, verbose=False)
                result = results[0]
                
                # 4. Handle Detections and Save Logic
                if len(result.boxes) > 0:
                    # conf=False removes the confidence percentage from the box
                    output_img = result.plot(conf=False) 
                    save_path = "defect_detected.jpg"
                    status = f"DEFECTS FOUND: {len(result.boxes)}"
                else:
                    output_img = frame
                    save_path = "no_defect_detected.jpg"
                    status = "NO DEFECTS DETECTED"
                
                # Save to disk
                cv2.imwrite(save_path, output_img)
                
                # 5. Display in Notebook
                rgb_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                
                clear_output(wait=True)
                print(f"Inspection Time: {time.strftime('%H:%M:%S')}")
                print(f"Status: {status}")
                print(f"Waiting {POST_PROCESS_WAIT} seconds for next PCB...")
                display(pil_img)
            
            else:
                print(f"Camera Error: HTTP {response.status_code}")

        except Exception as e:
            print(f"Network/Capture error: {e}")
            print("Retrying in 2 seconds...")
            time.sleep(2)
            continue

        # 6. The 5-second rest period happens HERE, after everything else is done
        time.sleep(POST_PROCESS_WAIT)

except KeyboardInterrupt:
    print("\nInspection system stopped.")
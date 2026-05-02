import cv2
from ultralytics import YOLO
import os

def run_pcb_inference(model_path, image_path):
    # 1. Load the YOLOv8 TFLite model
    # The 'task=detect' argument ensures it treats it as an object detection model
    model = YOLO(model_path, task='detect')

    # 2. Run inference
    # conf=0.25 is the standard threshold; adjust as needed for your PCB defects
    results = model(image_path, conf=0.25)
    
    # Extract the first result (since we only passed one image)
    result = results[0]
    
    # 3. Check if any defects were detected
    if len(result.boxes) > 0:
        print(f"Detected {len(result.boxes)} defect(s).")
        output_name = "defect_detected.jpg"
        
        # Plot the boxes on the image
        # result.plot() returns a BGR numpy array
        annotated_img = result.plot()
        cv2.imwrite(output_name, annotated_img)
    else:
        print("No defects detected.")
        output_name = "no_defect_detected.jpg"
        
        # Load the original image to save it with the new name
        original_img = cv2.imread(image_path)
        cv2.imwrite(output_name, original_img)

    print(f"Result saved as: {output_name}")

if __name__ == "__main__":
    # Ensure your tflite file and image path are correct
    run_pcb_inference('best.tflite', 'test.jpg')
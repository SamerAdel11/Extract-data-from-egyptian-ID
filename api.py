from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
from easyocr import Reader
import convert_numbers
reader = Reader(['ar'])
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
from datetime import datetime
app=FastAPI()
print("LOL")
# model = YOLO("bestAA.pt")
def extract_features(img,model):
    # Load the YOLO model

    # Load the image
    # img = cv2.imread(image_path)

    # Predict on the image
    results = model.predict(source=img, conf=0.25, imgsz=640)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Extract class ID
            class_name = result.names[class_id]  # Get the class name
            confidence = box.conf[0].item()  # Get the confidence score

            print(f"Class: {class_name}, Confidence: {confidence:.2f}")

    # Get the masks from the results
    masks = results[0].masks

    # Ensure masks exist
    if masks is not None and len(masks.data) > 0:
        # Convert the first mask tensor to a NumPy array
        first_mask = masks.data[0].cpu().numpy()

        # Convert the mask to a binary image (0 and 255)
        mask = (first_mask * 255).astype('uint8')

        # Resize the mask to match the original image size
        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Find contours in the resized mask
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the convex hull of the largest contour
            hull = cv2.convexHull(largest_contour)

            # Approximate the hull to a polygon and ensure it has 4 vertices
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) == 4:
                # Sort the points to get a consistent order: top-left, top-right, bottom-right, bottom-left
                pts = np.array([point[0] for point in approx], dtype="float32")
                sums = pts.sum(axis=1)
                diffs = np.diff(pts, axis=1)

                top_left = pts[np.argmin(sums)]
                bottom_right = pts[np.argmax(sums)]
                top_right = pts[np.argmin(diffs)]
                bottom_left = pts[np.argmax(diffs)]

                # Define the destination points for the perspective transform
                width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
                height = max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_right))

                dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

                # Compute the perspective transform matrix and apply it
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_img = cv2.warpPerspective(img, matrix, (int(width), int(height)))

                # Correct orientation if necessary
                if class_name == "front-bottom":
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_180)
                elif class_name == "front-left":
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_CLOCKWISE)
                elif class_name == "front-right":
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif class_name == "back-bottom":
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_180)
                elif class_name == "back-left":
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_CLOCKWISE)
                elif class_name == "back-right":
                    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                return cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

    return None  # Return None if no valid feature extraction is done

def replace_punctuation(text):
    # Create a translation table to replace punctuation with a hyphen
    translation_table = str.maketrans(string.punctuation, '-' * len(string.punctuation))
    # Remove punction in the start and the end of the sentence
    text = text.strip(string.punctuation)
    # Replace punctuation with hyphen
    return text.translate(translation_table)

def clean_ocr_output(ocr_output):
    print("OCR FUnction")
    # # Arabic digit Unicode range: \u0660 to \u0669
    # arabic_digit_range = {chr(i) for i in range(0x0660, 0x066A)}

    # # Filter out only Arabic digits
    # cleaned_output = ''.join(char for char in ocr_output if char in arabic_digit_range)
    if all(char.isdigit() for char in ocr_output):
        print("text is digit",ocr_output)
        if len(ocr_output)<13:
            print("less than 13")
            return ocr_output
        elif not ocr_output.startswith(('٢','٣')) and len(ocr_output)>14:

            print("First char trimmed")
            return clean_ocr_output(ocr_output[1:])
        elif ocr_output.startswith(('٢','٣')) and len(ocr_output)>14:
            print("Last char trimmed")
            return clean_ocr_output(ocr_output[:len(ocr_output)-1])
        else:
            return ocr_output
    else:
        print("text is string")
        return replace_punctuation(ocr_output)

def split_id(image, detector, exclude_labels=['face','Add1','Add2']):
    if exclude_labels is None:
        exclude_labels = {'Face'}

    # Load the image
    # image = cv2.imread(image_path)
    print(image.shape)
    # Run inference
    results = detector.predict(image, conf=0.25, imgsz=640)
    # Access the first result
    result = results[0]
    # print("results is ",result)

    # Get bounding boxes, confidence scores, and class labels
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

    # Create the output folder if it doesn't exist
    images_dict=[]
    # Crop and save each detected object, excluding specified labels
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        # Get the label name
        label = result.names[int(class_id)]
        print(label)
        # Skip if the label is in the exclude list
        if label in exclude_labels:
            continue

        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, box)

        # Crop the image around the bounding box
        cropped_image = image[ymin:ymax, xmin:xmax]
        image_dict={"image":cropped_image,
        "label":label}
        images_dict.append(image_dict)
        # Create a filename for the cropped image
        # crop_filename = f'{label}.jpg'  # Same name for each label
        # crop_path = os.path.join(output_folder, crop_filename)
    return images_dict
        # Save the cropped image, overriding if it already exists
        # cv2.imwrite(crop_path, cropped_image)

detector = YOLO('best555.pt')
rotation_model = YOLO("bestAA.pt")

# output_folder = r'output_images'

@app.post("/extract_id/")
async def create_upload_file(file: UploadFile = File(...)):
    # Read the content of the file
    start_time=datetime.now()
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cropping_start=datetime.now()
    image=extract_features(image,rotation_model)
    cropping_end=datetime.now()
    diff=cropping_end-cropping_start
    image = cv2.resize(image,(600,400),interpolation=cv2.INTER_LANCZOS4)
    image = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=12, searchWindowSize=21)
    split_start=datetime.now()
    images_dict=split_id(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),detector)
    split_end=datetime.now()

    extracted_text={}
    for image in images_dict:
        extract_start=datetime.now()
        if image['label'] == 'Id':
            ocr_result=reader.recognize(image['image'],allowlist='٠١٢٣٤٥٦٧٨٩')
        else:
            ocr_result=reader.recognize(image['image'])
        extract_end=datetime.now()
        result_easy_ocr = sorted(ocr_result, key=lambda x: x[0][1])
        extracted_id=clean_ocr_output(''.join(l[1] for l in result_easy_ocr))
        label=image['label']
        extracted_text[label]=extracted_id
        extracted_text[f'{label}_time']=extract_end-extract_start
    extracted_text['cropping_time']=diff
    extracted_text['splitting_time']=split_end-split_start
    end_time=datetime.now()
    extracted_text['total_time']=end_time-start_time
    return extracted_text

@app.post("/")
def index(file: UploadFile = File(...)):
    return {"Message":"Hello"}
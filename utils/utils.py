import string
import numpy as np
import cv2

# =======================
# Text Processing Utilities
# =======================

def clean_text(text):
    """
    Cleans text by replacing punctuation with hyphens and stripping surrounding punctuation.
    Args:
        text (str): The input text to clean.
    Returns:
        str: The cleaned text.
    """
    translation_table = str.maketrans(string.punctuation, '-' * len(string.punctuation))
    text = text.strip(string.punctuation)
    return text.translate(translation_table)

def clean_id(extracted_id):
    """
    Cleans and validates an extracted ID based on specific rules.
    Args:
        extracted_id (str): The ID string to clean.
    Returns:
        str: The cleaned ID, or the original if it's valid.
    """
    if len(extracted_id) < 14:
        print("Extracted ID is Incorrect")
        return extracted_id
    elif not extracted_id.startswith(('٢', '٣')) and len(extracted_id) > 14:
        return clean_ocr_output(extracted_id[1:])
    elif extracted_id.startswith(('٢', '٣')) and len(extracted_id) > 14:
        return clean_ocr_output(extracted_id[:len(extracted_id) - 1])
    else:
        return extracted_id

def clean_ocr_output(ocr_output):
    """
    Cleans OCR output by determining whether it's numeric or text-based.
    Args:
        ocr_output (str): The OCR output to clean.
    Returns:
        str: The cleaned OCR output.
    """
    if all(char.isdigit() for char in ocr_output):
        return clean_id(ocr_output)
    else:
        return clean_text(ocr_output)


# =======================
# Image Processing Utilities
# =======================

def extract_text_from_images(images_dict, reader):
    """Extracts text from images using OCR."""
    extracted_text = {}
    for image in images_dict:
        label = image['label']
        extract_start = datetime.now()
        
        # If the coming segment is ID, we force the result to be numbers
        if label == 'Id':
            ocr_result = reader.recognize(image['image'], allowlist='٠١٢٣٤٥٦٧٨٩')
        else:
            ocr_result = reader.recognize(image['image'])
        
        extract_end = datetime.now()
        
        # The result may come in wrond order, so we sort the coming values by the the corresponding char box
        result_easy_ocr = sorted(ocr_result, key=lambda x: x[0][1])

        # The date might come with noise, so we remove that noise
        extracted_id = utils.clean_ocr_output(''.join(l[1] for l in result_easy_ocr))
        
        # Append the result and their time to the dictionary
        extracted_text[label] = extracted_id
        extracted_text[f'{label}_time'] = extract_end - extract_start
    
    return extracted_text

def preprocess_image(image):
    """
    Preprocesses an image by resizing and denoising.
    Args:
        image (np.ndarray): The input image.
    Returns:
        np.ndarray: The processed image.
    """
    image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_LANCZOS4)
    image = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=12, searchWindowSize=21)
    return image

def extract_features(img, model):
    """
    Extracts features from an image using a trained model.
    Args:
        img (np.ndarray): The input image.
        model (YOLO): The pre-trained YOLO model for feature extraction.
    Returns:
        np.ndarray or None: The processed image or None if no features are found.
    """
    results = model.predict(source=img, conf=0.25, imgsz=640)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Extract class ID
            class_name = result.names[class_id]  # Get the class name
            confidence = box.conf[0].item()  # Get the confidence score
            print(f"Class: {class_name}, Confidence: {confidence:.2f}")

    masks = results[0].masks
    if masks is not None and len(masks.data) > 0:
        first_mask = masks.data[0].cpu().numpy()
        mask = (first_mask * 255).astype('uint8')
        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) == 4:
                pts = np.array([point[0] for point in approx], dtype="float32")
                dst_pts, matrix = calculate_perspective_transform(pts, img.shape)
                warped_img = cv2.warpPerspective(img, matrix, (int(dst_pts[1][0]), int(dst_pts[2][1])))
                return cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    return None

def calculate_perspective_transform(pts, img_shape):
    """
    Calculates the perspective transform matrix for a set of points.
    Args:
        pts (np.ndarray): Array of points (4x2).
        img_shape (tuple): Shape of the original image.
    Returns:
        tuple: Destination points and transformation matrix.
    """
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]

    width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    height = max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(bottom_left - top_left))
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return dst_pts, matrix

def split_id_into_segments(image, detector, exclude_labels=['face']):
    """
    Splits an ID image into labeled segments using a YOLO detector.
    Args:
        image (np.ndarray): The input image.
        detector (YOLO): The pre-trained YOLO model for detection.
        exclude_labels (list): Labels to exclude from segmentation.
    Returns:
        list[dict]: A list of dictionaries with cropped image segments and labels.
    """
    results = detector.predict(image, conf=0.25, imgsz=640)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    images_dict = []
    for box, class_id in zip(boxes, class_ids):
        label = result.names[int(class_id)]
        if label in exclude_labels:
            continue
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_image = image[ymin:ymax, xmin:xmax]
        images_dict.append({"image": cropped_image, "label": label})
    return images_dict

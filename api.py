from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from ultralytics import YOLO
import cv2
import numpy as np
from easyocr import Reader

from datetime import datetime

from utils import utils

reader = Reader(['ar'])
app=FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load the models
detector = YOLO('split_image.pt')
rotation_model = YOLO("crop_and_rotate.pt")


@app.post("/extract_id")
async def create_upload_file(file: UploadFile = File(...)):
    start_time = datetime.now()

    # Step 1: Read file
    image = await read_file(file)

    # Step 2: Crop and extract features
    image, cropping_time = crop_and_extract_features(image, rotation_model)

    # Step 3: Preprocess image
    image = utils.preprocess_image(image)

    # Step 4: Split ID sections
    images_dict, splitting_time = split_id_cards(image, detector)

    # Step 5: Extract text
    extracted_text = utils.extract_text_from_images(images_dict, reader)

    # Step 6: Add timing information
    extracted_text['cropping_time'] = cropping_time
    extracted_text['splitting_time'] = splitting_time
    extracted_text['total_time'] = datetime.now() - start_time

    return extracted_text

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    # HTML page that allows users to upload an image
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    # Updated HTML page for the landing page
    return templates.TemplateResponse("landing_page.html",{"request":request})


async def read_file(file: UploadFile):
    """Reads the uploaded file and converts it to an image."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def crop_and_extract_features(image, rotation_model):
    """Crops and extracts features from the image using a rotation model."""
    start_time = datetime.now()
    processed_image = utils.extract_features(image, rotation_model)
    end_time = datetime.now()
    cropping_time = end_time - start_time
    return processed_image, cropping_time

def split_id_cards(image, detector):
    """Splits the ID card image into sections using a detector."""
    start_time = datetime.now()
    images_dict = utils.split_id_into_segments(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), detector)
    end_time = datetime.now()
    splitting_time = end_time - start_time
    return images_dict, splitting_time

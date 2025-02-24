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
import os

from utils import utils

reader = Reader(['ar'])
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

from functools import lru_cache

@lru_cache(maxsize=1)  # Caches the model in memory
def get_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    detector = YOLO(os.path.join(BASE_DIR, 'models/split_image.pt'))
    rotation_model = YOLO(os.path.join(BASE_DIR, "models/crop_and_rotate.pt"))
    return detector, rotation_model


@app.post("/extract_id")
async def create_upload_file(file: UploadFile = File(...)):
    start_time = datetime.now()
    detector,rotation_model=get_models()
    # Step 1: Read file 
    image = await read_file(file)

    # Step 2: Crop and extract features
    image = utils.extract_features(image, rotation_model)

    # Step 3: Preprocess image
    image = utils.preprocess_image(image)

    # Step 4: Split ID sections
    images_dict = utils.split_id_into_segments(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), detector)

    # Step 5: Extract text
    extracted_text = utils.ocr(images_dict, reader)

    # Step 6: Add timing information
    # extracted_text['cropping_time'] = cropping_time
    # extracted_text['splitting_time'] = splitting_time
    extracted_text['total_time'] = datetime.now() - start_time

    return extracted_text


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    # HTML page that allows users to upload an image
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    # Updated HTML page for the landing page
    return templates.TemplateResponse("landing_page.html", {"request": request})


async def read_file(file: UploadFile):
    """Reads the uploaded file and converts it to an image."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

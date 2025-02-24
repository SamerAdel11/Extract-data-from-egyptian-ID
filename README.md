# OCR System for ID Card Extraction and Component Segmentation

## 📌 Project Overview
This project is an Optical Character Recognition (OCR) system built with **FastAPI** to process ID cards by:
1. **Extracting the ID card from its background**.
2. **Segmenting the ID card into its components** (e.g., name, ID number, photo, etc.).
3. **Providing a web interface for easy interaction**.

The extracted components can be further processed for text recognition and data extraction.

---

## 📂 Project Structure
```
├── api.py          # FastAPI application
├── models/         # YOLO models
├── utils/          # Utility functions for processing
├── static/         # CSS, JavaScript, and other assets
├── templates/      # HTML templates for the frontend
├── requirements.txt # Required dependencies
└── README.md
```

---

## 🛠️ Installation & Setup
### Prerequisites
- Python 3.11
- FastAPI
- Uvicorn
- OpenCV
- YOLO
- EasyOCR

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/SamerAdel11/Extract-data-from-egyptian-ID
cd Extract-data-from-egyptian-ID

# Install dependencies
pip install -r requirements.txt
```

### Run the FastAPI Server
```bash
uvicorn api:app --reload
```

---

## 🚀 Usage
### 1️⃣ Access the Web Interface
After running the server, open your browser and go to:
```
http://127.0.0.1:8000
```

### 2️⃣ API Endpoints
#### 🔹 Extract ID from Image
```http
POST /extract_id/
```
**Request Body:**
- `file`: Image file of the ID card.

**Response:**
- JSON object containing extracted ID card data along with their processing time

#### 🔹 Extract Text from an ID Component
```http
POST /extract_text/
```
**Request Body:**
- `file`: Image of the specific ID component.

**Response:**
- Extracted text in JSON format.

---

### 📩 Contact
For any inquiries, feel free to reach out at sameradel789@gmail.com.

---

This README now includes the `/extract_id` endpoint and the web interface. Let me know if you need more refinements! 🚀


import os
import requests
import pandas as pd

# Configuration
DIRECTORY = os.path.expanduser("api\\accuracy\\Egyptian ID's dataset")
API_ENDPOINT = "http://localhost:9000/extract_id"
CSV_FILENAME = "api\\accuracy\\data - Copy.xlsx"

# Load the Excel file
df = pd.read_excel(CSV_FILENAME)

# List JPG files
jpg_files = sorted([f for f in os.listdir(DIRECTORY) if f.lower().endswith(".jpg")])
times = []

# Iterate over files in the directory
for filename in jpg_files:
    file_path = os.path.join(DIRECTORY, filename)

    with open(file_path, 'rb') as image_file:
        files = {'file': image_file}
        request = requests.post(API_ENDPOINT, files=files)
        if request.status_code==422:

            df.loc[
                (df['folder'] == int(filename.split('.')[0])),
                'extracted_text'
            ] = "Can't be proccessed"
            continue

        response = request.json()
        total_time = response.pop('total_time')
        for key, value in response.items():
            print("value is", value)
            print(f"folder name is {filename.split('.')[0]}, filename is {key}.jpg")
            # print("row is ", df.loc[
            #     (df['folder'] == int(filename.split('.')[0])) & 
            #     (df['filename'] == f'{key.strip()}.jpg'), 

            # ])
            df.loc[
                (df['folder'] == int(filename.split('.')[0])) & 
                (df['filename'] == f'{key.strip()}.jpg'), 
                'extracted_text_not_preproccessed'
            ] = value

        # break
# Try to save the Excel file to the intended location
try:
    output_path = "api/accuracy/accuracy.xlsx"
    df.to_excel(output_path)
    print(f"File saved to {output_path}")
except Exception as e:
    # Fallback to default directory (Downloads)
    fallback_dir = os.path.join(os.environ['USERPROFILE'], 'Downloads')
    fallback_path = os.path.join(fallback_dir, "accuracy_fallback.xlsx")
    try:
        df.to_excel(fallback_path)
        print(f"Original save failed: {e}")
        print(f"File saved instead to fallback directory: {fallback_path}")
    except Exception as fallback_error:
        print(f"Failed to save file even in fallback directory: {fallback_error}")

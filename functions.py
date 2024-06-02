import requests
import numpy as np
import cv2
import os
import shutil
from skimage import exposure,filters,color
from tensorflow.keras.models import load_model

def sendMail(target, subject, message): 
    url = "https://rapidprod-sendgrid-v1.p.rapidapi.com/mail/send"

    payload = {
        "personalizations": [
            {
                "to": [{"email": target}],
                "subject": subject
            }
        ],
        "from": {"email": "StockSensei@bot.ai"},
        "content": [
            {
                "type": "text/html",
                "value": message
            }
        ]
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "01859bbbdbmsh5ef4be697540182p16dee3jsnd363a79130f7",
        "X-RapidAPI-Host": "rapidprod-sendgrid-v1.p.rapidapi.com"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    
    return response


def preprocess_image(gray_image):
    
    # Ensure the input image is of the correct type for CLAHE
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_limit_image = clahe.apply(gray_image)
    # Gamma correction
    gamma_corrected_image = cv2.convertScaleAbs(contrast_limit_image, alpha=1.2)
    # Gaussian filtering
    filtered_image = filters.gaussian(gamma_corrected_image, sigma=0.5)
    # Reshape the image to have a single channel
    processed_image = filtered_image.reshape(filtered_image.shape + (1,))

    return processed_image


def Detect(image):
    image_size = (300, 300)
    image = image.astype(np.float32)
    image = cv2.resize(image, image_size)
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Binary_model.h5')
    proba = model.predict(image)
    return proba[0][0]     

def Detect_Segment_Microaneurysms(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/Microaneurysm_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    image = segmented.astype(np.float32)
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Binary_Microaneurysm_model.h5')
    proba = model.predict(image)
    
    return segmented,proba[0][0]

def Detect_Segment_Haemorrhages(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/Haemorrhages_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    image = segmented.astype(np.float32)
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Binary_Haemorrhages_model.h5')
    proba = model.predict(image)
    
    return segmented,proba[0][0]


def Detect_Segment_HardExudates(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/HardExudates_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    image = segmented.astype(np.float32)
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Binary_HardExudates_model.h5')
    proba = model.predict(image)
    
    return segmented,proba[0][0]


def Segment_Microaneurysms(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/Microaneurysm_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    return segmented

def Segment_Haemorrhages(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/Haemorrhages_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    return segmented


def Segment_HardExudates(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/HardExudates_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    return segmented


def Classifie(image):
    image_size = (300, 300)
    image = image.astype(np.float32)
    image = cv2.resize(image, image_size)
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Multi_model.h5')
    proba = model.predict(image)
    return proba[0]     

def Classifie_Segment_Microaneurysms(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/Microaneurysm_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    image = segmented.astype(np.float32)
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Multi_Microaneurysm_model.h5')
    proba = model.predict(image)
    
    return segmented,proba[0]

def Classifie_Segment_Haemorrhages(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/Haemorrhages_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    image = segmented.astype(np.float32)
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Multi_Haemorrhages_model.h5')
    proba = model.predict(image)
    
    return segmented,proba[0]


def Classifie_Segment_HardExudates(image):
    
    image = image.astype(np.float32)
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=0)
   
    segmenter = load_model('./static/models/HardExudates_u-net_preprocessed.h5')
    segmented = segmenter.predict(image)
    segmented = (segmented > 0.99).astype(np.uint8)
    segmented = np.squeeze(segmented)
    
    image = segmented.astype(np.float32)
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=0)
    
    model = load_model('./static/models/Multi_HardExudates_model.h5')
    proba = model.predict(image)
    
    return segmented,proba[0]

def clear_uploads():
    upload_dir = './static/uploads'

    # Remove all files and subdirectories in the upload directory
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
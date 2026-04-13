import numpy as np
import cv2
import math
import os
import csv
from pathlib import Path
from tqdm import tqdm
from derm_ita import get_fitzpatrick_type
from src.pipeline.skin_extraction import extract_patch
from utils.utils import get_paths, get_file_paths

def calc_ita(median_bgr):
    bgr_pixel = np.array([[median_bgr]], dtype=np.float32)
    bgr_normalized = bgr_pixel / 255.0
    lab_pixel = cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2LAB)
    
    L = lab_pixel[0, 0, 0]
    b = lab_pixel[0, 0, 2]
    
    if b == 0:
        b = 1e-5
        
    ita = math.atan((L - 50) / b) * (180 / math.pi)
    
    return ita

def get_monk_thresholds():
    monk_hex = [
        "#F6EDE4", "#F3E7DB", "#F7EAD0", "#EADABA", "#D7BD96",
        "#A07E56", "#825C43", "#604134", "#3A312A", "#292420"
    ]
    
    itas = []
    for h in monk_hex:
        h = h.lstrip('#')

        bgr = [int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]
        itas.append(calc_ita(bgr))
        
    thresholds = []
    for i in range(len(itas) - 1):
        midpoint = (itas[i] + itas[i+1]) / 2.0
        thresholds.append(midpoint)
        
    return thresholds

def get_monk_type(ita_value, thresholds):
    for i, thresh in enumerate(thresholds):
        if ita_value >= thresh:
            return f"scale {i+1:02d}"
    return "scale 10"

def format_fitz(classification):
    ROMAN_SCALE = {
        '1': 'type i', '2': 'type ii', '3': 'type iii', 
        '4': 'type iv', '5': 'type v', '6': 'type vi'
    }
    
    tone_roman = ROMAN_SCALE.get(str(classification), "N/A")
    return tone_roman

def run_ita(result_path, scale):
    columns = ['File', f'Tone label ({scale.capitalize()})', 'ITA Value']
    image_paths = get_paths()
    
    file_exists = os.path.exists(result_path)
    monk_thresholds = get_monk_thresholds()
    
    with open(result_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(columns)

        for path in tqdm(image_paths, desc=f"Calculating ITA ({scale})"):
            _, skin_path = get_file_paths(path)

            img_skin = cv2.imread(skin_path)
            
            if img_skin is None:
                print(f"Error: The image couldn't be read: {skin_path}")
                continue
            
            median_bgr, _ = extract_patch(img_skin)
            
            if median_bgr is not None:
                ita_value = calc_ita(median_bgr)
                
                if scale.lower() == 'monk':
                    tone_label = get_monk_type(ita_value, monk_thresholds)
                else:
                    fitz_type = get_fitzpatrick_type(ita_value)
                    tone_label = format_fitz(fitz_type)
                
                img_data = [
                    Path(path).name,
                    tone_label,
                    round(ita_value, 2)
                ]
            
                writer.writerow(img_data)
            else:
                print(f"Skin not found in {path}")
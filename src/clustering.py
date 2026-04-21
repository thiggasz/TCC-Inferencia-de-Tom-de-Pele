import subprocess
import os
import cv2
import csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from src.pipeline.skin_extraction import get_skin_pixels
from utils.utils import get_paths, get_file_paths

"""

Classify skin tone based on the the centroid of the skin pixels clustering

"""

def hex_to_lab(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0]
    return lab_pixel

def bgr_to_hex(bgr_array):
    b, g, r = np.round(bgr_array).astype(int)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

TONES = {
    "fitzpatrick": {
        "type i": ("#F4D0B1", hex_to_lab("#F4D0B1")),
        "type ii": ("#E7B48F", hex_to_lab("#E7B48F")),
        "type iii": ("#D29F7C", hex_to_lab("#D29F7C")),
        "type iv": ("#BA7851", hex_to_lab("#BA7851")),
        "type v": ("#A55E2B", hex_to_lab("#A55E2B")),
        "type vi": ("#3C1F1D", hex_to_lab("#3C1F1D"))
    },
    "monk": {
        "scale 01": ("#F6EDE4", hex_to_lab("#F6EDE4")),
        "scale 02": ("#F3E7DB", hex_to_lab("#F3E7DB")),
        "scale 03": ("#F7EAD0", hex_to_lab("#F7EAD0")),
        "scale 04": ("#EADABA", hex_to_lab("#EADABA")),
        "scale 05": ("#D7BD96", hex_to_lab("#D7BD96")),
        "scale 06": ("#A07E56", hex_to_lab("#A07E56")),
        "scale 07": ("#825C43", hex_to_lab("#825C43")),
        "scale 08": ("#604134", hex_to_lab("#604134")),
        "scale 09": ("#3A312A", hex_to_lab("#3A312A")),
        "scale 10": ("#292420", hex_to_lab("#292420"))
    }
}

def clusterize_skin(skin_pixels, scale, clusters=3):
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(skin_pixels)
    
    # Find the dominat cluster
    counts = np.bincount(kmeans.labels_)
    dominant_cluster_index = np.argmax(counts)
    dominant_color_bgr = kmeans.cluster_centers_[dominant_cluster_index]
    
    # Conversion to CIELAB
    dominant_bgr_img = np.uint8([[dominant_color_bgr]])
    dominant_lab = cv2.cvtColor(dominant_bgr_img, cv2.COLOR_BGR2LAB)[0][0]
    
    selected_scale = TONES[scale]
    min_distance = float('inf')
    closest_match = None
    
    # Compare the CIELAB value with the scale tones and classify skin tone
    for name, (hex_val, scale_lab) in selected_scale.items():
        # Usa o array LAB que já está pronto no dicionário!
        distance = np.linalg.norm(dominant_lab.astype(float) - scale_lab.astype(float))
        
        if distance < min_distance:
            min_distance = distance
            closest_match = name
            
    return closest_match, dominant_color_bgr

def run_clustering(scale, clusters=3, result_path='results_clustering.csv'):
    columns = ['file', 'tone label', 'dominat color']
    image_paths = get_paths()
    
    file_exists = os.path.exists(result_path)
    
    with open(result_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(columns)

        for path in tqdm(image_paths, desc=f"Calculating Clustering ({scale})"):
            face_path, _, mask_path = get_file_paths(path)
            
            img_face = cv2.imread(face_path)
            img_mask = cv2.imread(mask_path)
            
            if img_face is None:
                print(f"Error: The image couldn't be read: {face_path}")
                continue
            
            _, skin_pixels = get_skin_pixels(img_face, img_mask)
            
            tone_label, dominant_color = clusterize_skin(skin_pixels, scale, clusters)
            
            img_data = [
                Path(path).name,
                tone_label,
                bgr_to_hex(dominant_color)
            ]
        
            writer.writerow(img_data)
    
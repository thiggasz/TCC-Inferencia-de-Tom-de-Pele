import cv2
import os
from typing import List
from tqdm import tqdm
from src.pipeline.face_detection import FaceDetector

def find_good_image(images_p: List[str], pipeline: FaceDetector) -> str:
    images_p = sorted(images_p)
    
    for img_p in images_p:
        img = cv2.imread(img_p)
        if img is None:
            continue
            
        face_crop = pipeline.process_image(img)
        
        if face_crop is not None:
            return img_p
            
    return ""

def select_faces():
    input_dirs_txt = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\paths\folders\complete.txt"
    output_txt = "filtered.txt"
    
    with open(input_dirs_txt, 'r') as file:
        directories = [line.strip() for line in file if line.strip()]

    pipeline = FaceDetector()
    files = []
    
    for subject_path in tqdm(directories, desc="Processing folder"):
        images = os.listdir(subject_path)
        images_p = [os.path.join(subject_path, p) for p in images]

        scripted = [p for p in images_p if "_scripted" in p]
        nonscripted = [p for p in images_p if "_nonscripted" in p]

        good_nonscripted = find_good_image(nonscripted, pipeline)
        good_scripted = find_good_image(scripted, pipeline)

        if good_nonscripted:
            files.append(good_nonscripted)
            print(f"\nCould not find face 'nonscripted': {subject_path}")

        if good_scripted:
            files.append(good_scripted)
        elif len(scripted) > 0:
            print(f"\nCould not find face 'scripted': {subject_path}")
            
        if len(nonscripted) == 0 and len(scripted) == 0:
            print(f"\nEmpty folder: {subject_path}")

    
    with open(output_txt, "w") as f:
        for file in files:
            f.write(file + "\n")

if __name__ == "__main__":
    select_faces()
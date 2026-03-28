import cv2
import os
import numpy as np
from tqdm import tqdm
import gc

from pipeline.face_detection import FaceDetector
from pipeline.face_segmentation import FaceSegmenter

def get_paths():
    input_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\paths\filtered.txt"
    
    if not os.path.exists(input_file):
        print(f"{input_file} not found.")
        return
        
    with open(input_file, 'r') as file:
        image_paths = [line.strip() for line in file if line.strip()]
        
    return image_paths
    
def get_skin_masks(output_folder):
    batch_size = 16                       
    
    os.makedirs(output_folder, exist_ok=True)
    
    pipeline = FaceDetector()
    segmenter = FaceSegmenter(batch_size=batch_size)
    image_paths = get_paths()
        
    total_images = len(image_paths)
    print(f"Total images: {total_images}")
    
    for i in tqdm(range(0, total_images, batch_size), desc="Batch Processing"):
        lote_paths = image_paths[i : i + batch_size]
        
        valid_crops = []
        valid_names = []
        
        for path in lote_paths:
            img = cv2.imread(path)
            if img is None:
                continue
                
            face_crop = pipeline.process_image(img)
            if face_crop is not None:
                valid_crops.append(face_crop)
                
                parts = os.path.normpath(path).split(os.sep)
                safe_name = "_".join(parts[-1:])
                safe_name = safe_name.replace(":", "")
                safe_name = os.path.splitext(safe_name)[0] 
                
                valid_names.append(safe_name)
                
        if valid_crops:
            masks, skins = segmenter.batched_deep_segmentation(valid_crops)
            
            for name, skin_mask in zip(valid_names, skins):
                if skin_mask.max() <= 1:
                    skin_mask = (skin_mask * 255).astype(np.uint8)
                    
                save_path = os.path.join(output_folder, f"skin_{name}.png")
                cv2.imwrite(save_path, skin_mask)

if __name__ == "__main__":
    OUTPUT_FOLDER = r"C:\Users\thiag\Dataset CCv2\Maks"
    
    get_skin_masks(OUTPUT_FOLDER)
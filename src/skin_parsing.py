import cv2
import os
import numpy as np
from tqdm import tqdm
from utils.utils import get_paths
from pipeline.face_detection import FaceDetector
from pipeline.face_segmentation import FaceSegmenter

def detect_faces(face_folder, skin_folder):
    batch_size = 16                       
    
    os.makedirs(face_folder, exist_ok=True)
    os.makedirs(skin_folder, exist_ok=True)
    
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
                
                face_path = os.path.join(face_folder, f"{safe_name}_face.png")
                cv2.imwrite(face_path, face_crop)
                
        if valid_crops:
            masks, skins = segmenter.batched_deep_segmentation(valid_crops)
            
            for name, skin_mask in zip(valid_names, skins):
                if skin_mask.max() <= 1:
                    skin_mask = (skin_mask * 255).astype(np.uint8)
                    
                skin_path = os.path.join(skin_folder, f"{name}_skin.png")
                cv2.imwrite(skin_path, skin_mask)

if __name__ == "__main__":
    OUTPUT_FACE = r"C:\Users\thiag\Dataset CCv2\Faces"
    OUTPUT_SKIN = r"C:\Users\thiag\Dataset CCv2\Masks"
    OUTPUT_PATCH = r"C:\Users\thiag\Dataset CCv2\Patchs"
    
    detect_faces(OUTPUT_FACE, OUTPUT_SKIN)
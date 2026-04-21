import cv2
import os
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from src.pipeline.face_detection import FaceDetector
from utils.utils import get_paths, get_file_paths

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

detector = FaceDetector()

def get_skin_luminance(img_face, img_mask):
    img_lab = cv2.cvtColor(img_face, cv2.COLOR_BGR2LAB)
    luminance = img_lab[:, :, 0]
    
    mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    is_skin = mask_gray > 127
    skin_pixels = luminance[is_skin]
    
    if skin_pixels.size == 0:
        return None, None

    return float(np.mean(skin_pixels)), float(np.std(skin_pixels))

def get_skin_sharpness(img_face, img_mask):
    face_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
    
    mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    is_skin = mask_gray > 127
    
    laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
    laplacian_values = laplacian[is_skin]
    
    if len(laplacian_values) == 0:
        return None

    sharpness_score = np.var(laplacian_values)
    
    return sharpness_score

def get_face_pose(img, file):
    face_crop = detector.get_face_crop(img)
    
    if face_crop is None:
        print(f"A face crop was not detected in: {file}")
        return None, None, None
    
    crop_h, crop_w, _ = face_crop.shape

    crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    detected_face = face_mesh.process(crop_rgb)
    
    if not detected_face.multi_face_landmarks:
        print(f"A face mesh was not detected in: {file}")
        return None, None, None
        
    face_landmarks = detected_face.multi_face_landmarks[0]
    
    points_2d = np.array([
        (face_landmarks.landmark[1].x * crop_w, face_landmarks.landmark[1].y * crop_h),
        (face_landmarks.landmark[152].x * crop_w, face_landmarks.landmark[152].y * crop_h),
        (face_landmarks.landmark[33].x * crop_w, face_landmarks.landmark[33].y * crop_h),
        (face_landmarks.landmark[263].x * crop_w, face_landmarks.landmark[263].y * crop_h),
        (face_landmarks.landmark[61].x * crop_w, face_landmarks.landmark[61].y * crop_h),
        (face_landmarks.landmark[291].x * crop_w, face_landmarks.landmark[291].y * crop_h)
    ], dtype="double")
    
    points_3d = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye
        (225.0, 170.0, -135.0),      # Right eye
        (-150.0, -150.0, -125.0),    # Left corner of the mouth
        (150.0, -150.0, -125.0)      # Right corner of the mouth
    ])
    
    focal_distance = crop_w
    center = (crop_w / 2, crop_h / 2)
    camera_matrix = np.array([
        [focal_distance, 0, center[0]],
        [0, focal_distance, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    distorcion_coef = np.zeros((4, 1))
    
    sucess, rotation_vector, translation_vector = cv2.solvePnP(
        points_3d, points_2d, camera_matrix, distorcion_coef, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not sucess:
        print(f"Perspective was not solved in: {file}")
        return None, None, None
        
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    
    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    
    if pitch < -90: 
        pitch += 180
    elif pitch > 90: 
        pitch -= 180
    
    if roll < -90: 
        roll += 180
    elif roll > 90: 
        roll -= 180

    return yaw, pitch, roll

def get_skin_noise(img_face, img_mask):
    kernel_erosion = np.ones((5,5), np.uint8)
    mask_eroded = cv2.erode(img_mask, kernel_erosion, iterations=2)
    
    face_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
    
    mask_gray = cv2.cvtColor(mask_eroded, cv2.COLOR_BGR2GRAY)
    is_skin = mask_gray > 127
    
    kernel_noise = np.array([[ 1, -2,  1],
                             [-2,  4, -2],
                             [ 1, -2,  1]], dtype=np.float64)
    
    noise_map = cv2.filter2D(face_gray, cv2.CV_64F, kernel_noise)
    
    skin_noise_values = noise_map[is_skin]
    
    if len(skin_noise_values) == 0:
        return None
        
    score_noise = np.std(skin_noise_values)
    
    return score_noise

def get_contrast(img_face, img_mask):    
    face_lab = cv2.cvtColor(img_face, cv2.COLOR_BGR2LAB)
    luminance = face_lab[:, :, 0]
    
    mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    is_skin = mask_gray > 127
    
    skin_lum = luminance[is_skin]
    background_lum = luminance[~is_skin]
    
    if len(skin_lum) == 0 or len(background_lum) == 0:
        return None
        
    contrast = np.mean(skin_lum) - np.mean(background_lum)
    
    return contrast

def get_temperature_and_spill(img_face, img_mask):
    kernel = np.ones((3, 3), np.uint8)
    mask_erode = cv2.erode(img_mask, kernel, iterations=1)

    mask_gray = cv2.cvtColor(mask_erode, cv2.COLOR_BGR2GRAY)
    is_skin = mask_gray > 127

    face_LAB = cv2.cvtColor(img_face, cv2.COLOR_BGR2LAB)
    channel_a = face_LAB[:, :, 1]
    channel_b = face_LAB[:, :, 2]

    skin_a = channel_a[is_skin]
    skin_b = channel_b[is_skin]

    if skin_a.size == 0:
        return None, None

    return float(np.mean(skin_a)), float(np.mean(skin_b))

def analyse_dataset(result_path):
    columns = ['file', 'luminance mean', 'luminance std', 'noise', 'sharpness', 'yaw', 'pitch', 'roll', 'contrast', 'spill', 'temperature']
    image_paths = get_paths()
    
    file_exists = os.path.exists(result_path)
    
    with open(result_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(columns)

        for path in tqdm(image_paths, desc="Analysing dataset"):
            face_path, _, mask_path = get_file_paths(path)
            img = cv2.imread(path)
            img_face = cv2.imread(face_path)

            img_mask = cv2.imread(mask_path)
            
            mean, std = get_skin_luminance(img_face, img_mask)
            sharpness = get_skin_sharpness(img_face, img_mask)
            noise = get_skin_noise(img_face, img_mask)
            yaw, pitch, roll = get_face_pose(img, path)
            contrast = get_contrast(img_face, img_mask)
            spill, temp = get_temperature_and_spill(img_face, img_mask)
            
            img_data = []
            img_data.append(Path(path).name)
            img_data.append(mean)
            img_data.append(std)
            img_data.append(noise)
            img_data.append(sharpness)
            img_data.append(yaw)
            img_data.append(pitch)
            img_data.append(roll)
            img_data.append(contrast)
            img_data.append(spill)
            img_data.append(temp)
            
            writer.writerow(img_data) 
     
if __name__ == "__main__":
    OUTPUT_FILE = "ccv2_analysis.csv"
    
    analyse_dataset(OUTPUT_FILE)
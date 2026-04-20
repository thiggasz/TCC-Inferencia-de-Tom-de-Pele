import cv2
import numpy as np

def get_skin_pixels(segmented_bgr, skin_mask):
    """Given a image of segmentated skin, returns the median BGR color and a skin pixels array"""
    
    # Obtain the skin pixels based on the mask
    is_skin = np.all(skin_mask == [255, 255, 255], axis=-1) 
    skin_pixels_bgr = segmented_bgr[is_skin]
    
    if len(skin_pixels_bgr) == 0:
        return None, None
        
    # Convert the skin pixels to CIELAB
    skin_pixels_bgr_reshaped = skin_pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    skin_pixels_lab = cv2.cvtColor(skin_pixels_bgr_reshaped, cv2.COLOR_BGR2LAB)
    
    luminance = skin_pixels_lab[:, 0, 0]
    
    # Disregard pixels with extreme luminaces
    lower_bound = np.percentile(luminance, 15)
    upper_bound = np.percentile(luminance, 90)
    
    filtered_maks = (luminance >= lower_bound) & (luminance <= upper_bound)
    selected_skin_pixels = skin_pixels_bgr[filtered_maks]

    if len(selected_skin_pixels) == 0:
        selected_skin_pixels = skin_pixels_bgr

    # Calculate the median BGR value
    median_bgr = np.median(selected_skin_pixels, axis=0)
    
    return median_bgr, selected_skin_pixels
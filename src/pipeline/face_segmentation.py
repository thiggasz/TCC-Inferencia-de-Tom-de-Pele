import torch
from src.libs.FaceParser.inference import Segmentator
from typing import Tuple, List, Generator
from tqdm import tqdm
import numpy as np

"""""

Given a imagem crop of a detected face, segmentates the face in the image, excluding the protected atributes,
resulting in a skin pixels mask

""""" 

class FaceSegmenter:
    def __init__(self, batch_size: int = 32):
        self.segmentator = Segmentator()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.segmentator.model.to(self.device)
        
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: Using {gpu_name} for segmentation.")
        else:
            print("Aviso: GPU not detected. Using CPU for segmentation.")
            
        self.batch_size = batch_size

    def deep_segmentation(self, image: np.array) -> Tuple[np.array, np.array]:
        return self.segmentator.inference(image)

    def batch_list(self, data: List[np.array]) -> List[List[np.array]]:
        return [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]

    def batched_deep_segmentation(self, images: List[np.array]) -> Tuple[List[np.array], List[np.array]]:
        batches = self.batch_list(images)
        masks = []
        skins = []
        
        for batch in tqdm(batches, desc="Segmentando batches"):
            m, sk = self.segmentator.batched_inference(batch)
            masks.extend(m)
            skins.extend(sk)
            
        return masks, skins
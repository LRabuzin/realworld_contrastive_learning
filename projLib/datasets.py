import os
import pandas as pd
import torch
from torchvision.datasets.folder import pil_loader
from typing import Dict, Optional
from PIL import Image

class RealWorldIdentDataset(torch.utils.data.Dataset):
    """Pytorch Dataset used for fetching real world data from the TAO dataset
    according to a given partitioning per latents
    """
    def __init__(self, data_dir : str, image_pairs: pd.DataFrame, has_labels: bool = True, transform: Optional[torch.nn.Module] = None) -> None:
        """Initializes the instance based on the given image pairs and transform

        The expected format of the image pair file is a csv file containing
        the columns "image1", "image2" and "content" where "image1" and
        "image2" are paths to the files containing corresponding images and
        "content" is a single class value (liable to change)

        Args:
        data_dir: directory_containing images
        image_pairs_filepath: path to file containing image pair locations
        has_labels: boolean indicating whether the dataset contains labels
        transform: torchvision.transforms Transform
        """
        self.data_dir = data_dir
        self.image_pairs = image_pairs
        self.has_labels = has_labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Optional[Image.Image|str]]:
        image_1_path = self.image_pairs['image1'][idx]
        image_1 = pil_loader(os.path.join(self.data_dir, image_1_path))
        if self.transform is not None:
            image_1 = self.transform(image_1)
        
        image_2_path = self.image_pairs['image2'][idx]
        image_2 = pil_loader(os.path.join(self.data_dir, image_2_path))
        if self.transform is not None:
            image_2 = self.transform(image_2)

        if self.has_labels:
            z = self.image_pairs['content'][idx]
        else:
            z = None

        return {
            "image1": image_1,
            "image2": image_2,
            "content": z
        }
    
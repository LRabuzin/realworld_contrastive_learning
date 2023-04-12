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
    def __init__(self,
                 data_dir : str,
                 image_pairs: pd.DataFrame,
                 partition: str = "train",
                 has_labels: bool = True,
                 transform: Optional[torch.nn.Module] = None,
                 keep_in_memory: bool = True,
                 ava_and_hacs_present: bool = False) -> None:
        """Initializes the instance based on the given image pairs and transform

        The expected format of the image pair file is a csv file containing
        the columns "image1", "image2" and "content" where "image1" and
        "image2" are paths to the files containing corresponding images and
        "content" is a frozenset of content categories

        Args:
        data_dir: directory_containing images
        image_pairs: path to file containing image pair locations
        has_labels: boolean indicating whether the dataset contains labels
        transform: torchvision.transforms Transform
        keep_in_memory: boolean indicating whether the images should be
        kept in memory. Setting to False slows down execution, but reduces
        memory requirements.
        """
        self.data_dir = data_dir
        self.image_pairs = image_pairs
        self.has_labels = has_labels
        self.transform = transform
        self.keep_in_memory = keep_in_memory
        self.ava_and_hacs_present = ava_and_hacs_present
        self.labels=[]

        if self.keep_in_memory:
            self.images1 = []
            self.images2 = []
            for i, row in image_pairs.iterrows():
                if not self.ava_and_hacs_present and ("HACS" in row["image1"] or "AVA" in row["image1"] or "HACS" in row["image2"] or "AVA" in row["image2"]):
                    continue
                self.images1.append(pil_loader(os.path.join(self.data_dir,"TAO_frames", "frames", partition, row["image1"])))
                self.images2.append(pil_loader(os.path.join(self.data_dir,"TAO_frames", "frames", partition, row["image2"])))
                self.labels.append(row["content"])
    
    def __len__(self) -> int:
        return len(self.images1)
    
    def __getitem__(self, idx: int) -> Dict[str, Image.Image|Optional[str]]:
        if self.keep_in_memory:
            image_1 = self.images1[idx]
            image_2 = self.images2[idx]
        else:
            image_1_path = self.image_pairs['image1'][idx]
            image_1 = pil_loader(os.path.join(self.data_dir, image_1_path))
            image_2_path = self.image_pairs['image2'][idx]
            image_2 = pil_loader(os.path.join(self.data_dir, image_2_path))

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        if self.has_labels:
            z = self.labels[idx]
        else:
            z = None

        return {
            "image1": image_1,
            "image2": image_2,
            "content": list(z)
        }
    
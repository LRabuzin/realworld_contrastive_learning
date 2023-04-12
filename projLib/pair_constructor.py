import itertools
import json
from typing import Optional, Union, Tuple, Iterable, Dict
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .utils import find_occurences

class PairConfiguration:
    def __init__(self,
            label_json_paths: list[str],
            categories_json_path: str,
            count_instances: bool = False,
            k: Optional[int] = None,
            n: Optional[Union[int, list[int]]] = None,
            content_categories: Optional[Iterable[int]] = None,
            style_categories: Optional[Iterable[int]] = None,
            prefixes: list[str] = ["train", "val", "test"]
    ) -> None:
        self.label_json_paths = label_json_paths
        self.k = k
        self.n = n
        if self.n == self.n and type(self.n) != list:
            self.n = [self.n]
        if self.n == self.n:
            self.n = set(self.n)
        self.count_instances = count_instances
        self.prefixes = prefixes
        
        self.categories_json_path = categories_json_path
        self.categories_decoder = self._load_categories()

        self.raw_labels, self.categories = self._load_labels()

        if content_categories is None and style_categories is None:
            self.content_categories, self.style_categories = self._find_content_and_style_categories()
        elif content_categories is not None and style_categories is not None:
            self.content_categories, self.style_categories = np.array(list(content_categories)), np.array(list(style_categories))
        else:
            print("content")
            print(content_categories)
            print("style")
            print(style_categories)
            raise RuntimeError("Content and style categories aren't both set.")

        self.valid_labels = self._filter_labels()

        self.combo_to_image = self._map_content_combinations_to_images()

    def sample_pairs(self):
        data = []
        for content_combo, images in self.combo_to_image.items():
            if len(images) < 2:
                continue
            current_images = images.copy()
            random.shuffle(current_images)
            for image_i in range(0, len(current_images), 2):
                if image_i == len(current_images) -1:
                    continue
                row = [current_images[image_i], current_images[image_i+1], content_combo]
                data.append(row)
        df = pd.DataFrame(columns=["image1", "image2", "content"], data=data)
        return df
    
    def _load_labels(self) -> Tuple[pd.DataFrame, list[int]]:
        all_dfs = []
        all_categories = []
        for label_path, prefix in zip(self.label_json_paths, self.prefixes):
            with open(label_path,'r') as f:
                data = json.loads(f.read())
            df_freeform = pd.json_normalize(data, record_path =['sequences'])

            df_freeform = df_freeform.explode(['annotated_image_paths','segmentations'])
            df_freeform = df_freeform.reset_index(drop=True)

            def discover_image_categories(row):
                track_to_category = {}
                n_tracks = len(list(filter(lambda t: "track_category_ids." in t, df_freeform.columns)))
                for i in range(1, n_tracks+1):
                    if row[f"track_category_ids.{i}"] == row[f"track_category_ids.{i}"]:
                        track_to_category[i] = row[f"track_category_ids.{i}"]
                categories = []
                for key in row["segmentations"]:
                    categories.append(track_to_category[int(key)])
                return categories

            df_freeform["list_object"]=df_freeform.apply(discover_image_categories, axis=1)
            all_categories.extend(list(itertools.chain.from_iterable(list(df_freeform["list_object"].values))))

            df_freeform["prefix"] = prefix

            all_dfs.append(df_freeform)

        final_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        return final_df, all_categories

    
    def _load_categories(self) -> Dict[int, str]:
        with open(self.categories_json_path, "r") as f:
            raw_categories = json.load(f)
        categories_decoder = {category["id"]: category["synset"].split(".")[0] for category in raw_categories}
        return categories_decoder
    
    def _find_content_and_style_categories(self) -> Tuple[np.ndarray, np.ndarray]:
        unique_categories, category_frequencies = find_occurences(self.categories)

        if self.k is not None:
            content_categories = unique_categories[-self.k:]
        else:
            content_categories = unique_categories
        style_categories = np.setdiff1d(unique_categories, content_categories)
        return content_categories, style_categories
    
    def _count_content_objects(self) -> pd.Series:
        if self.count_instances:
            return self.raw_labels["list_object"].apply(lambda x: sum(1 for k in x if k in set(self.content_categories)))
        else:
            return self.raw_labels["list_object"].apply(lambda x: sum(1 for k in set(x) if k in set(self.content_categories)))
        
    def _filter_labels(self) -> pd.DataFrame:
        self.raw_labels["n_content_objects"] = self._count_content_objects()
        if self.n == self.n:
            mask = self.raw_labels["n_content_objects"].apply(lambda x: x in self.n)
            return self.raw_labels[mask]
        else:
            return self.raw_labels
    
    def _map_content_combinations_to_images(self) -> Dict[frozenset[int], list[str]]:
        self.valid_labels.loc[:,"content_combo"] = self.valid_labels["list_object"].apply(lambda x: frozenset(filter(lambda t: t in self.content_categories, x)))
        combo_to_image = {}
        for _, row in self.valid_labels.iterrows():
            combo_to_image.setdefault(row["content_combo"], []).append(os.path.join(row["prefix"], row["dataset"], row["seq_name"], row["annotated_image_paths"]))
        return combo_to_image
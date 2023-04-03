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
            label_json_path: str,
            categories_json_path: str,
            count_instances: bool = False,
            k: Optional[int] = None,
            n: Optional[int] = None,
            class_restriction: Optional[set[int]] = None
    ) -> None:
        self.label_json_path = label_json_path
        self.k = k
        self.n = n
        self.count_instances = count_instances
        self.class_restriction = class_restriction
        
        self.categories_json_path = categories_json_path
        self.categories_decoder = self._load_categories()

        self.raw_labels, self.categories = self._load_labels()

        self.content_categories, self.style_categories = self._find_content_and_style_categories()

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
        with open(self.label_json_path,'r') as f:
            data = json.loads(f.read())
        df_freeform = pd.json_normalize(data, record_path =['sequences'])

        df_freeform = df_freeform.explode(['annotated_image_paths','segmentations'])
        df_freeform = df_freeform.reset_index(drop=True)

        def discover_image_categories(row):
            track_to_category = {}
            for i in range(1,11):
                if row[f"track_category_ids.{i}"] == row[f"track_category_ids.{i}"]:
                    track_to_category[i] = row[f"track_category_ids.{i}"]
            categories = []
            for key in row["segmentations"]:
                categories.append(track_to_category[int(key)])
            return categories

        df_freeform["list_object"]=df_freeform.apply(discover_image_categories, axis=1)

        categories=list(itertools.chain.from_iterable(list(df_freeform["list_object"].values)))

        return df_freeform, categories

    
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

        if self.n:
            mask = self.raw_labels["n_content_objects"].apply(lambda x: x == self.n)
            return self.raw_labels[mask]
        else:
            return self.raw_labels
    
    def _map_content_combinations_to_images(self) -> Dict[frozenset[int], list[str]]:
        self.valid_labels["content_combo"] = self.valid_labels["list_object"].apply(lambda x: frozenset(filter(lambda t: t in self.content_categories, x)))
        combo_to_image = {}
        for _, row in self.valid_labels.iterrows():
            combo_to_image.setdefault(row["content_combo"], []).append(os.path.join(row["dataset"], row["seq_name"], row["annotated_image_paths"]))
        return combo_to_image

class PairConstructor:
    def __init__(
            self,
            label_json_path: str,
            categories_json_path: str,
            count_instances: bool = False,
            k: Optional[int] = None,
            n: Optional[int] = None,
            class_restriction: Optional[set] = None
        ) -> None:
        self.label_json_path = label_json_path
        self.k = k
        self.n = n
        self.class_restriction = class_restriction
        self.categories_json_path = categories_json_path
        self.categories_decoder = self._load_categories()

        self.labels_df, self.classes = self._load_labels()
        self.labels_df["seq_identifier"] = self.labels_df["dataset"]+"/"+self.labels_df["seq_name"]
        self.sorted_class_values, self.sorted_class_frequencies = self._find_most_frequent_classes()

        if self.k == self.k:
            self.content_classes = self.sorted_class_values[-k:]
        else:
            self.content_classes = self.sorted_class_values
        self.style_classes = np.setdiff1d(self.sorted_class_values, self.content_classes)

        if count_instances:
            self.labels_df["no_of_content_objects"] = self.labels_df["list_object"].apply(lambda x: sum(1 for k in x if k in set(self.content_classes)))
        else:
            self.labels_df["no_of_content_objects"] = self.labels_df["list_object"].apply(lambda x: sum(1 for k in set(x) if k in set(self.content_classes)))
        
        if self.n:
            mask = self.labels_df["no_of_content_objects"].apply(lambda x: x == self.n)
            self.viable_images = self.labels_df[mask]
        else:
            self.viable_images = self.labels_df

        self.viable_classes = list(itertools.chain.from_iterable(list(self.viable_images["list_object"].apply(lambda x: list(frozenset(filter(lambda t: t in self.content_classes, x)))))))
        self.viable_content_classes = list(filter(lambda t: t in self.content_classes, self.viable_classes))
        self.viable_style_classes = list(filter(lambda t: t in self.style_classes, self.viable_classes))

        self.sorted_content_classes, self.sorted_content_class_frequencies = self._find_most_frequent_occurences(self.viable_content_classes)
        self.sorted_style_classes, self.sorted_style_class_frequencies = self._find_most_frequent_occurences(self.viable_style_classes)

        self.all_content_combinations = list(self.viable_images["list_object"].apply(lambda x: frozenset(filter(lambda t: t in self.content_classes, x))))
        self.sorted_content_combinations, self.sorted_content_combination_frequencies = self._find_most_frequent_content_combinations()

        # self.viable_images['content_combo'] = self.viable_images["list_object"].apply(lambda x: frozenset(filter(lambda t: t in self.content_classes, x)))
        # self.combo_to_image = {}
        # for i, row in self.viable_images.iterrows():
        #     self.combo_to_image.setdefault(row['content_combo'], []).append(os.path.join(row["seq_identifier"], row["annotated_image_paths"]))

        # self.all_style_combinations = list(self.viable_images["list_object"].apply(lambda x: frozenset(filter(lambda t: t in self.style_classes, x))))
        # self.sorted_style_combinations, self.sorted_style_combination_frequencies = self._find_most_frequent_occurences(self.all_style_combinations)

        # self.all_backgrounds = list(self.viable_images["seq_identifier"])
        # self.sorted_backgrounds, self.sorted_background_frequencies = self._find_most_frequent_occurences(self.all_backgrounds)

    def sample_pairs(self, n: int):
        data = []
        for content_combo, frequency in zip(self.sorted_content_combinations, self.sorted_content_combination_frequencies):
            if frequency < 2:
                continue
            current_images = list(self.combo_to_image[content_combo])
            random.shuffle(current_images)
            for image_i in range(0, len(current_images), 2):
                if image_i == len(current_images) -1:
                    continue
                row = [current_images[image_i], current_images[image_i+1], content_combo]
                data.append(row)
        df = pd.DataFrame(columns=["image1", "image2", "content"], data=data)
        return df


    def _load_categories(self):
        with open(self.categories_json_path, "r") as f:
            raw_categories = json.load(f)
        categories_decoder = {category["id"]: frozenset(category["synonyms"]) for category in raw_categories}
        return categories_decoder
    
    def _load_labels(self) -> Tuple[pd.DataFrame, list]:
        with open(self.label_json_path,'r') as f:
            data = json.loads(f.read())
        df_freeform = pd.json_normalize(data, record_path =['sequences'])

        df_freeform["number_images"]=df_freeform["annotated_image_paths"].apply(lambda x: len(x))

        df_freeform = df_freeform.explode(['annotated_image_paths','segmentations'])
        df_freeform = df_freeform.reset_index(drop=True)

        concat_func = lambda x: [int(x) 
                                 for x
                                 in [x[col]
                                    for col 
                                    in x.index
                                    if col.startswith('track_category_ids.')]
                                 if x == x and (not self.class_restriction or int(x) in self.class_restriction)]
        df_freeform["list_object"]=df_freeform.apply(concat_func, axis=1)

        df_freeform["nb_objects"]=df_freeform["segmentations"].apply(lambda x: len(list(x.keys())))
        classes=list(itertools.chain.from_iterable(list(df_freeform["list_object"].values)))

        return df_freeform, classes
    
    def _find_most_frequent_occurences(self, sequence: Iterable) -> Tuple[np.ndarray, np.ndarray]:
        values, frequencies = np.unique(sequence, return_counts=True)
        
        order=np.argsort(frequencies)[::1]

        sorted_values=values[order]
        sorted_frequencies=frequencies[order]
        return sorted_values, sorted_frequencies

    def _find_most_frequent_classes(self) -> Tuple[np.ndarray, np.ndarray]:
        values, frequencies = np.unique(self.classes, return_counts=True)
        
        order=np.argsort(frequencies)[::1]

        sorted_values=values[order]
        sorted_frequencies=frequencies[order]
        return sorted_values, sorted_frequencies
    
    def _find_most_frequent_content_combinations(self) -> Tuple[np.ndarray, np.ndarray]:
        values, frequencies = np.unique(self.all_content_combinations, return_counts=True)
        
        order=np.argsort(frequencies)[::1]

        sorted_values=values[order]
        sorted_frequencies=frequencies[order]
        return sorted_values, sorted_frequencies
    
    def plot_most_frequent_values(self, frequencies: Iterable, values: Iterable, title: str = "", log_scale: bool = True, color_red: int = 20) -> None:
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=frequencies,x=np.arange(len(values)), color = ['blue']*(len(values)-color_red) + ['red']*color_red)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        plt.show()

    def plot_viable_content_class_distribution(self, title: str = "") -> None:
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=self.sorted_content_class_frequencies,x=[next(iter(self.categories_decoder[content_class])) for content_class in self.sorted_content_classes])
        ax.set_title(title)
        # ax.set_yscale('log')
        plt.show()

    def plot_viable_style_class_distribution(self, title: str = "") -> None:
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=self.sorted_style_class_frequencies,x=np.arange(len(self.sorted_style_classes)))
        ax.set_title(title)
        # ax.set_yscale('log')
        plt.show()

    def plot_class_distribution(self, title: str = "") -> None:
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=self.sorted_class_frequencies,x=np.arange(len(self.sorted_class_values)))
        ax.set_title(title)
        # ax.set_yscale('log')
        plt.show()

    def plot_content_object_number(self, title: str = "") -> None:
        counts, frequencies = np.unique(self.labels_df["no_of_content_objects"], return_counts=True)

        order =np.argsort(counts)[::1]

        sorted_counts = counts[order]
        sorted_frequencies = frequencies[order]

        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=sorted_frequencies,x=sorted_counts)
        ax.set_title(title)
        plt.show()

    def plot_most_frequent_content_combinations(self, title: str = "", log_scale: bool = True, color_red: int = 805) -> None:
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=self.sorted_content_combination_frequencies,x=np.arange(len(self.sorted_content_combinations)), color = ['blue' if color_red not in combo else 'red' for combo in self.sorted_content_combinations])
        ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        plt.show()

    def plot_most_frequent_style_combinations(self, title: str = "", log_scale: bool = True, color_red: int = 20) -> None:
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=self.sorted_style_combination_frequencies,x=np.arange(len(self.sorted_style_combinations)), color = ['blue']*(len(self.sorted_style_combinations)-color_red) + ['red']*color_red)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        plt.show()

    def plot_co_occurence_content_style(self, remove_zero_columns = True):
        co_occ_matrix = pd.DataFrame(0, index=self.sorted_content_combinations, columns=self.style_classes)
        for i in range(len(self.all_style_combinations)):
            for style_class in self.all_style_combinations[i]:
                co_occ_matrix[style_class][self.all_content_combinations[i]] += 1
        if remove_zero_columns:
            co_occ_matrix = co_occ_matrix.loc[:, (co_occ_matrix != 0).any(axis=0)]
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.matshow(co_occ_matrix)
        plt.show
        return list(co_occ_matrix.columns)

    def plot_single_content_style_co_occurence(self, remove_zero_columns = True):
        co_occ_matrix = pd.DataFrame(0, index=self.content_classes, columns=self.style_classes)
        for i in range(len(self.all_style_combinations)):
            for style_class in self.all_style_combinations[i]:
                for content_class in self.all_content_combinations[i]:
                    co_occ_matrix[style_class][content_class] += 1
        if remove_zero_columns:
            co_occ_matrix = co_occ_matrix.loc[:, (co_occ_matrix != 0).any(axis=0)]
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.matshow(co_occ_matrix)
        plt.show()
        print(co_occ_matrix.columns)

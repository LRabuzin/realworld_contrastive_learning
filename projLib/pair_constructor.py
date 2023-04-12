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
            style_categories: Optional[Iterable[int]] = None
    ) -> None:
        self.label_json_paths = label_json_paths
        self.k = k
        self.n = n
        if self.n == self.n and type(self.n) != list:
            self.n = [self.n]
        if self.n == self.n:
            self.n = set(self.n)
        self.count_instances = count_instances
        
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
        for label_path in self.label_json_paths:
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

        if self.n:
            mask = self.raw_labels["n_content_objects"].apply(lambda x: x in self.n)
            return self.raw_labels[mask]
        else:
            return self.raw_labels
    
    def _map_content_combinations_to_images(self) -> Dict[frozenset[int], list[str]]:
        self.valid_labels.loc[:,"content_combo"] = self.valid_labels["list_object"].apply(lambda x: frozenset(filter(lambda t: t in self.content_categories, x)))
        combo_to_image = {}
        for _, row in self.valid_labels.iterrows():
            combo_to_image.setdefault(row["content_combo"], []).append(os.path.join(row["dataset"], row["seq_name"], row["annotated_image_paths"]))
        return combo_to_image


def num_pairs_same_bg(config: PairConfiguration) -> int:
    df = config.sample_pairs()
    df["is_same_bg"] = df.apply(lambda row: os.path.dirname(row["image1"]) == os.path.dirname(row["image2"]), axis=1)
    return df["is_same_bg"].sum()

def num_sampled_pairs(config: PairConfiguration) -> int:
    df = config.sample_pairs()
    return len(df)

def plot_distribution_of_content_classes(config: PairConfiguration, log_scale = False):
    content_categories = config.content_categories
    category_frequencies = []
    for category in content_categories:
        contains = config.valid_labels["content_combo"].apply(lambda t: category in t)
        category_frequencies.append(int(contains.sum()))
    
    order=np.argsort(category_frequencies)

    sorted_categories = content_categories[order]
    sorted_frequencies = sorted(category_frequencies)

    names = [config.categories_decoder[category] for category in sorted_categories]

    fig, ax = plt.subplots(1,figsize=(10,10))
    y_pos = range(len(names))
    ax.bar(height=sorted_frequencies, x=y_pos)
    plt.xticks(y_pos, names, rotation=90)
    if log_scale:
        ax.set_yscale('log')
    plt.show()
        
def plot_distribution_of_number_of_content_objects(config: PairConfiguration):
    numbers_of_content_objects = list(config.raw_labels["n_content_objects"])
    unique_counts, frequencies = np.unique(numbers_of_content_objects, return_counts=True)
    fig, ax = plt.subplots(1,figsize=(10,10))
    ax.bar(height=frequencies, x=unique_counts, label=frequencies)
    rects = ax.patches
    for rect, label in zip(rects, frequencies):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        )
    plt.show()


def plot_distribution_of_number_of_content_combinations(config: PairConfiguration, emphasize_class: float = None):
    combos = list(config.combo_to_image.keys())
    print(len(combos))
    frequencies = [len(value) for value in config.combo_to_image.values()]
    
    zipped = list(zip(combos, frequencies))
    
    zipped = sorted(zipped, key = lambda t: t[1])
    
    combos, frequencies = zip(*zipped)
    combos = list(combos)
    frequencies = list(frequencies)
    if emphasize_class is not None:
        color_array = ["tab:blue" if emphasize_class not in combo else "tab:red" for combo in combos]
    else:
        color_array = ["tab:blue"] * len(combos)

    fig, ax = plt.subplots(1,figsize=(18,10))
    ax.bar(height=frequencies, x=[6 * pos for pos in range(len(combos))], width = 3, color=color_array)
    rects = ax.patches
    for rect, label in zip(rects, frequencies):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        )
    ax.axes.xaxis.set_visible(False)
    plt.show()

def plot_distribution_of_style_classes(config):
    style_classes = list(itertools.chain.from_iterable(list(config.valid_labels["list_object"].apply(lambda x: list(set(filter(lambda t: t in config.style_categories, x)))))))
    unique_style, frequencies = np.unique(style_classes, return_counts=True)

    order=np.argsort(frequencies)

    unique_style = unique_style[order]
    frequencies = frequencies[order]

    fig, ax = plt.subplots(1,figsize=(20,10))
    ax.bar(height=frequencies, x=[4*pos for pos in range(len(unique_style))], label=frequencies, width=3)
    rects = ax.patches
    for rect, label in zip(rects, frequencies):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        )
    ax.axes.xaxis.set_visible(False)
    print(len(unique_style))
    plt.show()
    return unique_style

def plot_number_of_distinct_content_classes_co_occuring_with_style(config: PairConfiguration, style):
    contents = []
    counts = []
    for style in style:
        co_occuring_content = config.valid_labels["list_object"].apply(lambda t: [] if style not in t else list(set(filter(lambda x: x in config.content_categories, t))))
        contents.append(set(itertools.chain(*co_occuring_content)))
        counts.append(len(contents[-1]))

    order = np.argsort(counts)

    styles = config.style_categories[order]
    frequencies = sorted(counts)

    fig, ax = plt.subplots(1,figsize=(20,10))
    ax.bar(height=frequencies, x=range(len(styles)), label=frequencies)
    ax.xaxis.set_visible(False)
    print(len(styles))
    plt.show()


















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

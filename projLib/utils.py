from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pair_constructor import PairConfiguration
import itertools
import os

def find_occurences(sequence: Iterable) -> Tuple[np.ndarray, np.ndarray]:
        values, frequencies = np.unique(sequence, return_counts=True)
        
        order=np.argsort(frequencies)[::1]

        sorted_values=values[order]
        sorted_frequencies=frequencies[order]
        return sorted_values, sorted_frequencies

def plot_occurences(
              self, 
              frequencies: Iterable, 
              values: Iterable, 
              title: str = "", 
              log_scale: bool = True, 
              emphasize: Optional[Callable[[any], bool]] = None
            ) -> None:
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.bar(height=frequencies,x=values, color = ['red' if emphasize(value) else "blue" for value in values])
        ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        plt.show()

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
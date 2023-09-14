from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_occurences(sequence: Iterable) -> Tuple[np.ndarray, np.ndarray]:
        values, frequencies = np.unique(sequence, return_counts=True)
        
        order=np.argsort(frequencies)[::1]

        sorted_values=values[order]
        sorted_frequencies=frequencies[order]
        return sorted_values, sorted_frequencies

def plot_occurences(
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

# plot the balanced accuracies of the models, multiple models can be plotted at once
# also shows the mean balanced accuracy of the models as a line in the bar plots
def balanced_accuracies_bar_plot(results, names, config, title):
    results_dfs = []
    for result in results:
        results_dfs.append(pd.read_csv(result).set_index("metric"))

    results_ba_content=[] 
    for result in results_dfs:
        results_ba_content.append(result.iloc[5, 1:21])

    print(results_ba_content[-1])
    
    category_names_dirty = [config.categories_decoder.get(int(col)) for col in list(results_dfs[0].columns[1:21])]
    category_names=[]
    for name in category_names_dirty:
        category_names.append(name.replace("_", " ").capitalize())
    
    base_model_scores = {
        name: [score for score in list(result)][:10] for name, result in zip(names, results_ba_content)
    }

    base_model_scores_full = {
        name: [score for score in list(result)][:20] for name, result in zip(names, results_ba_content)
    }

    x = np.arange(0, 20, 2)  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0

    fig, (ax1, ax2) = plt.subplots(2,1,layout='constrained')
    fig.set_size_inches(12,7)
    colors = ["indianred", "green", "blue", "goldenrod", "purple"]
    for i, batch in enumerate(base_model_scores.items()):
        attribute, measurement = batch
        offset = width * multiplier
        rects = ax1.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        ax1.axhline(np.array(base_model_scores_full[attribute]).mean(), color=colors[i], linewidth=1, linestyle='--')
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Balanced accuracy')
    ax1.set_title(title)
    ax1.set_xticks(x + width, category_names[:10])
    ax1.legend(loc='upper left', ncols=2)
    ax1.set_ylim(0, 1)

    base_model_scores = {
        name: [score for score in list(result)][10:] for name, result in zip(names, results_ba_content)
    }

    x = np.arange(0, 20, 2)  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0

    for i, batch in enumerate(base_model_scores.items()):
        attribute, measurement = batch
        offset = width * multiplier
        rects = ax2.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        ax2.axhline(np.array(base_model_scores_full[attribute]).mean(), color=colors[i], linewidth=1, linestyle='--')
        multiplier += 1

    ax2.set_ylabel('Balanced accuracy')
    ax2.set_xticks(x + width, category_names[10:])
    ax2.set_ylim(0, 1)

    plt.show()

def balanced_accuracies_boxplot(results, names, title, x_axis_label):
    results_dfs = []
    for result in results:
        results_dfs.append(pd.read_csv(result).set_index("metric"))

    fig, ax = plt.subplots()
    bp = ax.boxplot([df.loc["balanced_acc"][1:21] for df in results_dfs], positions=[3,9,15, 21], patch_artist=True, labels=names, widths = 3)
    colors = ["indianred", "green", "blue", "goldenrod"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    fig.set_size_inches(7,7)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel(x_axis_label)
    ax.set_title(title)
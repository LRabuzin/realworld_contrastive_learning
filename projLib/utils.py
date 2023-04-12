from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

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
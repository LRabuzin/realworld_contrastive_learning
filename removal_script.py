import os
import pandas as pd
import sys

if __name__ == "__main__":
    partition = sys.argv[1]
    imgs = pd.read_csv("../annotated_images.csv")
    imgs = imgs["0"]
    imgs = set(imgs)
    print(imgs[0])
    for path, subdirs, files in os.walk(partition):
        for name in files:
            filename = os.path.join(path, name)
            if filename not in imgs:
                print()
                print()
                print(filename)
                os.remove(filename)
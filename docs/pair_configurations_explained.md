# Defining Content and Style in Contrastive Learning with TAO Dataset

## 1. Background:
To determine the efficacy of contrastive learning in pinpointing shared content in realistic contexts, we utilized an encoder model for extracting latent factors from images in the TAO dataset. Given that we are using real-world data, where control over the data generation process is absent, ground truth labels present in our dataset serve as the foundation for defining content and style latent factors.

## 2. Defining Content:
- In our experiments, we delineated content by zeroing in on the twenty most common classes within the TAO dataset. These predominant classes form the basis for our definition of content.
- The content of an image was defined by the presence or absence of these twenty classes, regardless of how many times a class appeared within an image. In other words, frequency didn't impact content delineation.
- Images were selected based on their inclusion of objects from these twenty most-represented classes. Those devoid of any such classes were excluded.

## 3. Defining Style:
- After segregating content, the residual classes in the TAO dataset were set as style classes.
- The style of an image was then determined by identifying which of these residual classes were present in the image.

# Using the PairConfiguration class for defining content and style

The PairConfiguration class defined in the `projLib/pair_configuration.py` file is designed to facilitate the creation and management of pairs of images from the TAO dataset.

## Explanation of class constructor arguments
- label_json_paths: List of paths to JSON files containing label information for images (this refers to the `train.json`, `val.json` and `test.json` files in the `frames` folder).
- categories_json_path: Path to the JSON file containing information about categories in the dataset (this refers to the `categories.json` file coming from the BURST dataset annotations).
- count_instances: Specifies whether the number n will be referring to the number of object instances or categories.
- k: Specifies the number of the most frequent categories that will be considered as content.
- n: Specifies the number of content or instance categories present in an image pair.
- content_categories and style_categories: These arguments allow for explicit specification of which categories are to be treated as content and style. If these are provided, k is ignored.

If the user doesn't provide explicit content_categories and style_categories, the class will use the k most frequent categories from the dataset as content categories. The remaining categories will be treated as style categories.

By changing `n`, we can define how many content categories will be in each image pair. Leaving `n` as unlimited produces a more complex setting than constraining `n` to e.g. 1 or 2, as there will be more content categories in any given image pair for the model to recognise.

By changing `k`, we can define how many of the most frequent categories will be treated as content. This is useful agian for changing the degree of complexity of the setting. If `k` is set to 1, the model will only have to recognise one category as content, which is a much simpler task than recognising e.g. 5 categories as content.

The `count_instances` parameter can be used to switch between counting categories and instances. If `count_instances` is set to True, the model will count the number of instances of each category in an image pair. If `count_instances` is set to False, the model will count the number of categories in an image pair. In our experiments, we have not considered this factor of variation, but it has been implemented to facilitate future exploration.

## Image sampling

The current implementation of the PairConfiguration class samples images from the TAO dataset as pairs without replacement.

Additionally, the train, test and validation splits present in the TAO dataset are not adhered to as the split sizes are impractical for our purposes. Instead, the entire dataset is merged and then split into 4 parts, which are then used as train, validation, test and heldout splits.
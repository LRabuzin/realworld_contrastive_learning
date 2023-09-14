# Identifiability of shared content factors in real-world multi-modal datasets

This repository contains the code for the paper "Identifiability of shared content factors in real-world multi-modal datasets".

## Getting started

### 0. Prerequisites
Set up a virtual environment and install the required packages. The code was tested with Python 3.10.9. A conda environment with the correct python version and required packages can be installed using the following command:
```
conda create --name <envname> --file requirements.txt
```

### 1. Download the datasets
The dataset used in this repo is the [BURST dataset](https://github.com/Ali2500/BURST-benchmark), which is based on the [TAO dataset](https://motchallenge.net/tao_download.php). The BURST dataset essentially consists of annotations for the TAO dataset, where the image frames actually come only from TAO. To gain access to the full TAO data, you will have to register an account with the authors of the TAO dataset.

The unzipped dataset contains many frames which don't have annotations. The unannotated files can be removed by calling `removal_script.py`.

The unzipped dataset will have the following structure:
```
frames
--train
    --ArgoVerse
    --BDD
    --Charades
    --LaSOT
    --YFCC100M
--val
...
```

The removal script can then be called on each individual folder in the `train`, `val` and `test` folders:
```
python removal_script.py frames/train/ArgoVerse
```

Download the annotations from the BURST dataset and place the `train.json`, `val.json` and `test.json` files in the `frames` folder.

### 3. Running experiments

The experiments can be run using the `main.py` script. Importantly, training an encoder model and evaluating downstream performance require separate calls to `main.py`. The script uses wandb to log metrics, to be able to use that functionality, you need to be logged into your wandb account.

The script takes the following arguments:
- `data_dir`: path to directory containing frames from the TAO dataset
- `model_dir`: directory where trained model will be/is saved
- `var_name`: suffix added to model that is being trained/evaluated
- `model_id`: name of directory where model is saved, if left empty, a random identifier will be generated
- `encoding_size`: length of the encoding trained by the model
- `hidden_size`: number of hidden layers in the final fully-connected model layer
- `encoder_number`: which iteration of the model to use (model is saved multiple times during training, when evaluating, we can choose which iteration to evaluate)
- `k`: number of most commonly occurring categories considered content
- `n`: number of allowed content categories per image pair
- `leq_content_factors`: determines whether number of allowed content categories can be less than n (if set to false, model will be trained only using pairs with exactly n content categories)
- `tau`: tau parameter used by the infonce loss
- `lr`: encoder model learning rate
- `batch_size`: batch size used to train encoder model
- `train_steps`: how many steps to train the encoder model for
- `log_steps`: determines logging frequency
- `val_steps`: determines validation frequency
- `checkpoint_steps`: determines checkpointing frequency
- `evaluate`: if set, will not train the model, only evaluate it
- `seed`: seed for random number generators
- `workers`: number of workers for dataparallel training
- `no_cuda`: if set, model will be trained/evaluated on cpu
- `save_all_checkpoints`: if set, model will be saved on every training step
- `load_from_storage`: if set, dataloader will not keep data in memory
- `use_pretrained_rn`: if set, encoder will be initialized with weights pretrained on imagenet dataset
- `default_weights`: if used when evaluating model, will not attempt to load a trained model and will evaluate untrained model
- `full_eval_steps`: determines frequency of running an evaluation of learned embeddings on downstream classification of content and style classes from images
- `use_simclr_head`: add simclr-style projection head on top of encoder model
- `projection_dim`: dimension of simclr-style projected embeddings
- `color_jitter_strength`: strength of color jitter used in data augmentation
- `use_logreg_for_eval`: use logistic regression for evaluation (instead of MLP)
- `use_gp_for_eval`: use gaussian process for evaluation (instead of MLP)
- `use_svc_for_eval`: use support vector classification for evaluation (instead of MLP)
- `only_eval_content`: only evaluate content classification (not style)
- `augment_eval`: augment evaluation data with additional data
- `use_rn34`: use resnet34 instead of resnet18 as encoder model
- `use_clip`: use CLIP instead of resnet as encoder model

Further usage instructions can be found in the docs directory.

## Repository structure
projLib:
    - datasets.py: contains the code to load the datasets
    - models.py: contains the definition of a simple downstream classifier model
    - utils.py: contains utility functions and functions to plot results of experiments
    - losses.py: contains the definition of the infonce loss functions used in the experiments
    - infinite_iterator.py: contains the definition of an infinite iterator used to iterate over the dataset
    - pair_configuration.py: contains the definition of the pair configuration used in the experiments, the pair configuration object is used for constructing pairs of samples from the dataset based on a  definition of what makes up a pair of images
    - main.py: contains the code to run the experiments
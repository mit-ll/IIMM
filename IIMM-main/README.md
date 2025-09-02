# Inter-Intra Modality Measure for Vision-Language Contrastive Encoders
Official implementation for the ICCV 2025 submission "The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models."

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2025 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

## Overview
### Requirements
Experiments were run on an Anaconda 2023b environment with CUDA 11.8.
- Python 3.9.15
- PyTorch 2.0.1
- NumPy 1.24.3
- Pandas 1.5.3
- SciPy 1.10.0
- Scikit-learn 1.2.1
- wilds 2.0.0
- [Open-CLIP-Torch](https://github.com/mlfoundations/open_clip) 2.24.0

To install, read the necessary packages from requirements.txt using pip. Then, manually install the torch-scatter and torch-geometric packages, which are needed by wilds.

```console
$ pip install -r requirements.txt
$ pip install torch_geometric torch_scatter
```

### Datasets

The following datasets are supported in this code base:
- Stanford Cars
- CIFAR100
- DTD
- EuroSAT
- [fMoW](https://github.com/p-lambda/wilds/)
- GTSRB
- [ImageNet](https://www.image-net.org/download)
- [ImageNetV2](https://github.com/modestyachts/ImageNetV2), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples) (for evaluation only)
- MNIST
- RESISC45
- STL10
- SUN397
- SVHN

All of the listed datasets except for Stanford Cars, RESISC45, ImageNet, and ImageNet's variants can be downloaded automatically by running `python download_datasets.py`. Unwanted dataset downloads can be commented out within the file. Downloaded datasets will be stored under `data/`.

Unfortunately, the original link to download Stanford Cars via PyTorch has broken since our initial research. However, you can manually download [Stanford Cars from Kaggle](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars?resource=download). Download and unzip the file in `data/`.

The download for RESISC45 can be found [here](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbWdLWXpBUkJsNWNhM0hOYUhJbHpwX0lYanM&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&o=OneUp). Download and unzip the file into the pre-created dataset subdirectory `data/resisc45`, where you will find the associated splits already located.

Links have been provided above for the manual downloads of ImageNet and its variants.

If different dataset locations (i.e. shared folders) are desired for particular datasets, please change the default value for the dataset class's `location` attribute within its associated dataset module.

To use one of these datasets within the scripts discussed in the [User Guide](#user-guide), provide one of the dataset options to the  `--dataset` parameter, if available.

### Supported PEFT Methods
- [CLIP-Adapter](https://arxiv.org/abs/2110.04544) (adapter)
- [LoRA](https://arxiv.org/abs/2106.09685) (lora)
- [BitFit](https://arxiv.org/abs/2106.10199) (bias)
- Attention-Layer Tuning (attention)
- Linear Probe (probe)

To use these PEFT methods within the scripts discussed in the [User Guide](#user-guide), provide the associated names in parentheses as input to the `--model-name` parameter when required. 

When using CLIP-Adapter, the model reduction can be set using the `--reduction` parameter, which has a default value of 4. For LoRA, the `--rank` parameter sets the rank to be used, with a default value of 16, and LoRA can optionally be added to the model's MLP layers by adding the `--lora-mlp` flag. For both BitFit and Attention-Layer Tuning, the number of transformer blocks in which to train either the bias or the attention layers can be specified using the `--train-blocks` parameter, with the default being all 12 transformer blocks in CLIP.

### Supported CLIP Model Variants
- CLIP (clip)
- SigLIP (siglip)
- CoCa (coca)
- CLIP-EVA-02 (eva02)

To use these models within the scripts discussed in the [User Guide](#user-guide), provide the associated names in parentheses as input to the `--model-name` parameter when required. 

## User Guide

### Obtaining Zero-Shot Accuracies and Embeddings
Any model can be tested zero-shot by following the testing procedure outlined in [Testing](#testing) and simply not passing an argument to the `--model-path` parameter.

To generate the image and text embeddings of a set of datasets using a given model, run the following python script:

```console
$ python get_embeddings.py --datasets <dataset name(s)> --model-name <model name> [--options]
```

The configurable options for this script are as follows:
| Optional Parameter       | Type    | Description | Default |
| ------------------------ | ------- | ------------| ------- |
| `--model-name`           | `str`   | The name of the model/fine-tuning method to be used | -- |
| `--model-path`           | `str`   | The filepath to the saved training weights of a selected model | -- |
| `--datasets`             | `str`   | Space-separated list of one or more of the datasets for which to get embeddings | -- |
| `--batch-size`           | `int`   | The number of samples in each batch | 128 |
| `--num-workers`          | `int`   | Number of workers to use for loading data | 4 |
| `--outfolder`            | `str`   | The path to the folder in which the embedding results file will be stored | "embeddings" |
| `--val-p`                | `float` | The percentage of the training data to be used as validation data (does not apply to fMoW) | 0.01 |
| `--reduction`            | `int`   | The reduction value to use when using CLIP-Adapter | 4 |
| `--rank`                 | `int`   | The rank to employ when using LoRA | 16 |
| `--lora-mlp`             | --      | If this option is included when using LoRA, LoRA is also applied to the MLP layers of the model | -- |

Again, to obtain the model's zero-shot embeddings, simply run the script without passing an argument to the `--model-path` parameter.

### Fine-Tuning
All models are fine-tuned using the `train.py` script. Vanilla CLIP can be fine-tuned end-to-end or via any of the supported PEFT methods, and any supported CLIP model variants can be trained end-to-end. The basic syntax is:

```console
$ python train.py --dataset [dataset name] --model-name [model name] [--options]
```

The training script's additional configurable options are:
| Optional Parameter       | Type    | Description | Default |
| ------------------------ | ------- | ------------| ------- |
| `--dataset`              | `str`   | The dataset to be used for training | -- |
| `--model-name`           | `str`   | The name of the model/fine-tuning method to be used | "clip" |
| `--model-dir`            | `str`   | The directory in which models will be stored during training | "ckpts" |
| `--bs`                   | `int`   | The number of samples in each batch | 128 |
| `--nepochs`              | `int`   | The number of training epochs to execute | 30 |
| `--lr`                   | `float` | Learning rate to be used by the optimizer | 1e-6 |
| `--wd`                   | `float` | Weight decay to be used by the optimizer | 1e-4 |
| `--num-workers`          | `int`   | Number of workers to use for loading data | 4 |
| `--data-dir`             | `str`   | Absolute or relative path pointing to the directory where the WILDS datasets can be found | "./data" |
| `--optimizer`            | `str`   | PyTorch optimizer to use, options are "Adam" or "SGD" | "SGD" |
| `--val-p`                | `float` | The percentage of the training data to be used as validation data (does not apply to fMoW) | 0.1 |
| `--reduction`            | `int`   | The reduction value to use when using CLIP-Adapter | 4 |
| `--rank`                 | `int`   | The rank to employ when using LoRA | 16 |
| `--lora-mlp`             | --      | If this option is included when using LoRA, LoRA is also applied to the MLP layers of the model | -- |
| `--train-blocks`         | `int`   | The number of CLIP's transformer blocks to train when using BitFit or Attention-Layer Tuning, out of 12 total | 12 |
| `--progress`             | --      | If this option is included, then a progress bar will be displayed when iterating over training or validation batches | -- |

### Testing
Any model can be tested on a set of one or more evaluation datasets at a time using the `test.py` script. The syntax is as follows:

```console
$ python test.py --model-name [model name] --eval-datasets [space-separated list of desired eval datasets] [--options]
```

Below is a table describing each of the configurable options for the test script.
| Optional Parameter   | Type  | Description                                                                                                          | Default |
| -------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- | ------- |
| `--model-name`       | `str` | The name of the model/fine-tuning method to be used | -- |
| `--model-path`       | `str` | The path to saved training weights for your model. If not provided, the chosen model will be tested as zero-shot     | -- |
| `--eval-datasets`    | `str` | Space-separated list of one or more of the datasets against which the model will be evaluated | -- |
| `--bs`               | `int` | The number of samples in each batch                                                                                  | 128 |
| `--num-workers`      | `int` | Number of workers to use for loading data                                                                            | 4 |
| `--reduction`        | `int`   | The reduction value to use when using CLIP-Adapter                                                                 | 4 |
| `--rank`             | `int`   | The rank to employ when using LoRA                                                                                 | 16 |
| `--lora-mlp`         | --      | If this option is included when using LoRA, LoRA is also applied to the MLP layers of the model                    | -- |
| `--train-blocks`     | `int`   | The number of CLIP's transformer blocks to train when using BitFit or Attention-Layer Tuning, out of 12 total      | 12 |
| `--progress`         | --    | If this option is included, then a progress bar will be displayed when iterating over training or validation batches | -- |
| `--outfile`          | `str` | The path at which the results file will be stored                                                                    | "results/results.csv" |

### Computing Transerability Metrics

After fine-tuning a model of interest and collecting the zero-shot and fine-tuned embeddings for the model, you can compute the transfer scores for each evaluated transferability metric using `get_transfer_scores.py`. The syntax is as follows:

```console
$ python get_transfer_scores.py --dataset [dataset_name] --eval-models [model name(s)] --metrics [metric name(s)]
```

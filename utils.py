from typing import List
from argparse import Namespace
import warnings

import numpy as np
import open_clip
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import weightedtau


def create_model_and_transforms(model_name: str):
    if model_name == 'siglip':
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:timm/ViT-B-16-SigLIP')
    elif model_name == 'coca':
        model, _, preprocess = open_clip.create_model_and_transforms(model_name='coca_ViT-B-32', pretrained='laion2B-s13B-b90k')
    elif model_name == 'eva02':
        warnings.filterwarnings('ignore', category=UserWarning)
        model, _, preprocess = open_clip.create_model_and_transforms(model_name='EVA02-B-16', pretrained='merged2b_s8b_b131k')
    else:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    return model, preprocess


def get_tokenizer(model_name: str):
    if model_name == 'siglip':
        tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
    elif model_name == 'coca':
        tokenizer = open_clip.get_tokenizer('coca_ViT-B-32')
    elif model_name == 'eva02':
        tokenizer = open_clip.get_tokenizer('EVA02-B-16')
    else:
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return tokenizer


def get_tokenized_captions(dataset_name, dataset_obj, tokenizer):

    caption_templates = {
        'Cars': lambda c: f'a photo of a {c}',
        'CIFAR100': lambda c: f'a photo of a {c}',
        'DTD': lambda c: f'a photo of a {c} texture',
        'EuroSAT': lambda c: f'a centered satellite photo of a {c}',
        'FMoW': lambda c: f'a satellite image of a {c}',
        'GTSRB': lambda c: f'a photo of a "{c}" traffic sign',
        'ImageNet': lambda c: f'a photo of a {c}',
        'ImageNetV2': lambda c: f'a photo of a {c}',
        'ImageNetSketch': lambda c: f'a photo of a {c}',
        'ImageNetRendition': lambda c: f'a photo of a {c}',
        'ImageNetAdversarial': lambda c: f'a photo of a {c}',
        'MNIST': lambda c: f'a photo of the number: "{c}"',
        'RESISC45': lambda c: f'satellite imagery of {c}',
        'STL10': lambda c: f'a photo of a {c}',
        'SUN397': lambda c: f'a photo of a {c}',
        'SVHN': lambda c: f'a photo of the number: "{c}"',
    }

    if "fmow" in dataset_name.lower():
        captions = [caption_templates["FMoW"](c) for c in fmow_list]
    else:
        captions = [caption_templates[dataset_name](c) for c in dataset_obj.classnames]

    tokenized_captions = tokenizer(captions)

    return tokenized_captions


def gen_mean(nums: List[float], p=1) -> float:
    # Calculates the generalized mean given a set of numbers and a parameter p
    # A value of p=1 is equivalent to computing the arithmetic mean
    return (sum([n**p for n in nums]) / len(nums))**(1/p)


def gensil_score(X, y, p_val=1, metric='cosine'):
    # See pairwise_distances documentation for list of supported values for metric
    X, y = np.array(X), np.array(y)
    dists = pairwise_distances(X, metric=metric)

    gs_scores = []
    for clust_id, point in zip(y, dists):
        intracluster_mask = y == clust_id

        # Calculate average intracluster distance
        intracluster_dists = point[(intracluster_mask) & (point != 0)]
        a = gen_mean(intracluster_dists, p_val)

        # Calculate minimum average intercluster distance
        b = np.inf
        for k in np.unique(y[~intracluster_mask]):
            intercluster_mask = y == k
            intercluster_dists = point[intercluster_mask]
            _b = gen_mean(intercluster_dists, p_val)
            if _b < b: b = _b
        gs_scores.append((b-a)/max(b,a))

    # Return the average GenSil score
    return gen_mean(gs_scores, p_val)


def pprint_args(args: Namespace):
    maxlen = 0
    for arg in vars(args):
        if len(arg) > maxlen:
            maxlen = len(arg)

    for arg in vars(args):
        print(f"    {arg:{maxlen}} : {getattr(args, arg)}")

    print("\n")


def unwrap_params(params):
    if 'clip_model.' in list(params.keys())[0]:
        new_params = {}
        for k, v in params.items():
            new_params[k.split('clip_model.')[1]] = v
        params = new_params
    elif 'model.' in list(params.keys())[0]:
        new_params = {}
        for k, v in params.items():
            new_params[k.split('model.')[1]] = v
        params = new_params

    return params


def w_kendall_metric(score_dict, finetune_gt_dict):
    unpacked_scores = score_dict.items()
    metric_scores = [a[1] for a in unpacked_scores]
    gts = []
    for a in unpacked_scores:
        gts.append(finetune_gt_dict[a[0]])
    tw_metric,_ = weightedtau(metric_scores, gts)
    return tw_metric


### Code from https://github.com/OpenGVLab/Multitask-Model-Selector ###
def iterative_A(A, max_iterations=3):
    '''
    calculate the largest eigenvalue of A
    '''
    x = A.sum(axis=1)
    #k = 3
    for _ in range(max_iterations):
        temp = np.dot(A, x) 
        y = temp / np.linalg.norm(temp, 2) 
        temp = np.dot(A, y)
        x = temp / np.linalg.norm(temp, 2)
    return np.dot(np.dot(x.T, A), y)
########################################################################
    
fmow_list = [
        "airport",
        "airport hangar",
        "airport terminal",
        "amusement park",
        "aquaculture",
        "archaeological site",
        "barn",
        "border checkpoint",
        "burial site",
        "car dealership",
        "construction site",
        "crop field",
        "dam",
        "debris or rubble",
        "educational institution",
        "electric substation",
        "factory or powerplant",
        "fire station",
        "flooded road",
        "fountain",
        "gas station",
        "golf course",
        "ground transportation station",
        "helipad",
        "hospital",
        "impoverished settlement",
        "interchange",
        "lake or pond",
        "lighthouse",
        "military facility",
        "multi-unit residential",
        "nuclear powerplant",
        "office building",
        "oil or gas facility",
        "park",
        "parking lot or garage",
        "place of worship",
        "police station",
        "port",
        "prison",
        "race_track",
        "railway bridge",
        "recreational facility",
        "road bridge",
        "runway",
        "shipyard",
        "shopping mall",
        "single-unit residential",
        "smokestack",
        "solar farm",
        "space facility",
        "stadium",
        "storage tank",
        "surface mine",
        "swimming pool",
        "toll booth",
        "tower",
        "tunnel opening",
        "waste disposal",
        "water treatment facility",
        "wind farm",
        "zoo",
    ]
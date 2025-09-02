# Testing script for evaluating catastrophic forgetting across datasets
import os
from collections import defaultdict
import argparse
from copy import deepcopy

import pandas as pd
import torch

from utils import pprint_args, create_model_and_transforms, get_tokenizer, unwrap_params
from models.wrappers import OpenCLIP, SigLip, CoCa
from models.clip import CLIPLinearProbe, CLIPAdapter, LoRACLIP
from train_helpers import model_eval, get_eval_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        choices=['adapter', 'attention', 'bias', 'lora', 'probe', 'clip', 'coca', 'eva02', 'siglip'],
        help='Optional CLIP model variants to use instead',
    )
    parser.add_argument(
        "--model-path",
        nargs='?',
        type=str,
        help='Path to the saved training weights for your model. Not providing path will use zero-shot CLIP',
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        choices=['Cars','CIFAR100','DTD','EuroSAT','FMoW-id','FMoW-ood','GTSRB','ImageNet','ImageNetV2','ImageNetSketch','ImageNetRendition','ImageNetAdversarial','MNIST','RESISC45','STL10','SUN397','SVHN'],
        help="Space-separated list of one or more of the datasets against which the model will be evaluated.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="./results/results.csv",
        help="Path to folder in which results file will be stored.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Shows progress during testing.",
    )
    parser.add_argument(
        "--train-blocks",
        nargs="?",
        type=int,
        help='If using the CLIP model variants "attention" or "bias", use this arg to specify the number of transformer blocks of which parameters are to be trained',
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank for LoRA models",
    )
    parser.add_argument(
        "--lora-mlp",
        action="store_true",
        help="Optional flag when using LoRA model to apply LoRA to MLP weights, in addition to the attention weights",
    )
    parser.add_argument(
        "--reduction",
        type=int,
        default=4,
        help="Reduction to use for CLIP-Adapter"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    pprint_args(args)

    # Load models and device
    print("Loading model..")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = create_model_and_transforms(args.model_name)
    tokenizer = get_tokenizer(args.model_name)

    if args.model_name == 'adapter':
        model = CLIPAdapter(clip_model=model, reduction=args.reduction)
    elif args.model_name == 'lora':
        model = LoRACLIP(clip_model=model, include_mlp=args.lora_mlp, rank=args.rank)
    elif args.model_name == 'probe':
        base_model = deepcopy(model)
        base_model.to(device)
    elif args.model_name == 'siglip':
        model = SigLip(model=model)
    elif args.model_name == 'coca':
        model = CoCa(coca_model=model)
    if args.model_name in ['attention', 'bias', 'clip', 'eva02']: model = OpenCLIP(model)
    model.to(device=device)
    
    if args.model_path and args.model_name != 'probe':
        params = torch.load(args.model_path)
        params = unwrap_params(params)
        model.load_state_dict(params, strict=True)
    elif args.model_path and args.model_name == 'probe':
        if len(args.eval_datasets) > 1:
            raise ValueError("Loaded linear probes can only be tested on the dataset on which they were trained. Please provide only that dataset to --eval-datasets and try again.")
        else:
            params = torch.load(args.model_path)
    else:
        params = None
        print("No model path provided. Using model as zero-shot.")

    # Gather test results
    results = defaultdict(list)
    
    for d in args.eval_datasets:
        test_loader, tokenized_captions = get_eval_loader(dataset=d,
                                                num_workers=args.num_workers,
                                                batch_size=args.bs,
                                                preprocess=preprocess,
                                                tokenizer=tokenizer)
        if args.model_name == 'probe':
            if 'fmow' in d.lower():
                n_classes = test_loader.dataset.n_classes
            elif d == 'GTSRB':
                n_classes = 43
            elif d == 'SVHN':
                n_classes = 10
            else:
                n_classes = len(test_loader.dataset.classes)
            model = CLIPLinearProbe(clip_model=base_model, n_classes=n_classes)
            if params: model.load_state_dict(params, strict=True)
            model.to(device)
        
        print(f"Evaluating {d}..")
        results[d].append(model_eval(model=model, device=device, loader=test_loader, tokenized_text=tokenized_captions, progress=args.progress))
        print(f"{results[d][0]:.2f}%")

    # Write results
    results_df = pd.DataFrame.from_dict(results)
    
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    results_df.to_csv(args.outfile, index=False)

    print("Done.")
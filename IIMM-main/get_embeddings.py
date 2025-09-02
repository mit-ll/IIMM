import os
import argparse

import numpy as np
import torch
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds import get_dataset as get_wilds_dataset
from wilds.common.data_loaders import get_train_loader as get_wilds_train_loader

from utils import (
    pprint_args, 
    unwrap_params, 
    create_model_and_transforms, 
    get_tokenizer, 
    get_tokenized_captions,
)
from train_helpers import get_eval_loader, maybe_dictionarize
from models.clip import CLIPAdapter, LoRACLIP
from models.wrappers import OpenCLIP, SigLip, CoCa
from datasets.registry import get_dataset

def get_embeddings(
        model: torch.nn.Module,
        device: torch.device,
        loader: torch.utils.data.DataLoader,
        tokenized_text: torch.Tensor,
        outfolder: str,
        model_name: str,
        dataname: str,
        zero_shot: bool,
):
    model.eval()
    
    dataset = "fmow" if isinstance(loader.dataset, WILDSSubset) else "other"
        
    image_save_path = os.path.join(outfolder, f"{model_name}_{dataname}_image_embeddings{'_zs' if zero_shot else ''}.csv")
    text_save_path = os.path.join(outfolder, f"{model_name}_{dataname}_text_embeddings{'_zs' if zero_shot else ''}.csv")
    
    with torch.no_grad():
        
        tokenized_text = tokenized_text.to(device=device)
        text_embeddings = model.encode_text(tokenized_text).detach().cpu().numpy()
        f=open(text_save_path, 'ab')
        np.savetxt(f, text_embeddings, fmt='%1.8f', delimiter=',')
        f.close()
        
        f=open(image_save_path, 'ab')
        for batch in loader:
            
            if dataset == "fmow":
                x, y, _ = batch
            else:
                batch = maybe_dictionarize(batch)
                x = batch['images']
                y = batch['labels']
            x = x.to(device=device)
            
            # Get embeddings
            image_embeddings = model.encode_image(x).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            # append label to image embedding as column
            out = np.vstack([image_embeddings.T, y]).T
            # append batch image embeddings out outfile
            np.savetxt(f, out, fmt='%1.8f', delimiter=',')
        f.close()
                    
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        choices=['Cars','CIFAR100','DTD','EuroSAT','FMoW-id','FMoW-ood','GTSRB','ImageNetV2','ImageNetSketch','ImageNetRendition','ImageNetAdversarial','MNIST','RESISC45','STL10','SUN397','SVHN'],
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--val-p",
        type=float,
        default=.01,
        help='Percentage of the training data to use as validation data during training.',
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="clip",
        choices=['adapter', 'attention', 'bias', 'lora', 'probe', 'clip', 'coca', 'eva02', 'siglip'],
        help='Optional CLIP model variants to use instead. No selection will use zero-shot CLIP',
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
    parser.add_argument(
        "--model-path",
        type=str,
        help='Path to the saved training weights for your model',
    )
    parser.add_argument(
        "--outfolder",
        type=str,
        default="./embeddings",
        help="Path to folder in which results file will be stored.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank for lora model",
    )
    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()
    pprint_args(args)

    print("Loading model..")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = create_model_and_transforms(args.model_name)
    tokenizer = get_tokenizer(args.model_name)

    ### Model Preparation ###
    if args.model_name == 'adapter':
        model = CLIPAdapter(clip_model=model, reduction=args.reduction)
    elif args.model_name == 'lora':
        model = LoRACLIP(clip_model=model, include_mlp=args.lora_mlp, rank=args.rank)
    elif args.model_name == 'siglip':
        model = SigLip(model=model)
    elif args.model_name == 'coca':
        model = CoCa(coca_model=model)
    else:
        model = OpenCLIP(clip_model=model)
        
    if args.model_path:
        params = torch.load(args.model_path)
        params = unwrap_params(params)
        model.load_state_dict(params=params, strict=True)
    else:
        print("No model path provided. Using model as zero-shot.")
    model.to(device=device)
    
    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)
        
    for d in args.datasets:
        print(f"Getting embeddings {d}..")
        # Data preparation
        if 'fmow' in d.lower():
            if 'ood' in d:
                subset = 'val'
            else:
                subset = 'train'
            fmow = get_wilds_dataset(dataset="fmow", download=False, root_dir="./data")
            data = fmow.get_subset(
                subset,
                # frac=0.1,
                transform=preprocess,
            )
            loader = get_wilds_train_loader(
                "standard",
                dataset=data,
                num_workers=args.num_workers,
                pin_memory=True,
                batch_size=args.batch_size,
            )
        else:
            if d in ['ImageNetV2', 'ImageNetSketch', 'ImageNetRendition', 'ImageNetAdversarial']:
                loader, tokenized_captions = get_eval_loader(dataset=d,
                                                num_workers=args.num_workers,
                                                batch_size=args.batch_size,
                                                preprocess=preprocess,
                                                tokenizer=tokenizer)
            else:
                dataset_name = d +'Val'

                data = get_dataset(dataset_name=dataset_name,
                                preprocess=preprocess, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                val_fraction=args.val_p, 
                                max_val_samples=5000)
                loader = data.train_loader
            
                # Create tokenized text captions
                tokenized_captions = get_tokenized_captions(d, data, tokenizer)
                
        get_embeddings(
            model = model,
            device = device,
            loader = loader,
            tokenized_text = tokenized_captions,
            outfolder = args.outfolder,
            model_name = args.model_name,
            dataname = d,
            zero_shot=False if args.model_path else True,
            )
            
    print("Done.")
        
        
import os
import argparse

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from utils import pprint_args
from metrics import gap, uniformity_image, uniformity_text, inter, IIMM

def metric_scores(data, model, root, metrics, outpath, df, prop=1):
    path = lambda d,e: os.path.join(root, f'{d}_{e}_embeddings.csv')
    images_df = pd.read_csv(path(data, 'image'), header=None) # last column of image embeddings df is class label
    texts_df = pd.read_csv(path(data, 'text'), header=None)
    images = normalize(images_df.iloc[:,:-1].values, axis=1) 
    text = normalize(texts_df.values, axis=1)
    labels = images_df.iloc[:,-1].to_numpy().astype(int)  
           
    if prop != 1:
        index = np.random.choice(len(images), size=int(len(images)*prop), replace=False)
        images = images[index]
        labels = labels[index]
            
    for k in metrics:
        if 'gap' in k:
            measure = metrics[k](images, text)
        elif k in ['cos_sim', 'uniformity_image', 'uniformity_text', 'IIMM']:
            measure = metrics[k](images, text, labels)
        elif k == 'inter_images':
            measure = metrics[k](images)
        elif k == 'inter_text':
            measure = metrics[k](text)
        else:
            X = np.vstack([images, text])
            Y = np.concatenate([np.ones(images.shape[0]), np.zeros(text.shape[0])])
            measure = metrics[k](X, Y)
        l = [data, model, prop, k, measure]
        _df = pd.DataFrame(l).T
        _df.columns = ['data', 'model', 'proportion', 'measure', 'score']
        df = pd.concat([df, _df])
    return df
    
def get_metric_scores(datasets, model, finetuning, root, metrics, outpath, prop):
    df = pd.DataFrame()
    for d in datasets:
        _df = metric_scores(d, model, root, metrics, outpath, df, prop)
        df = pd.concat([df, _df])
    df['finetuning'] = finetuning
    df.to_csv(outpath, index=False)
    return df
    
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=['clip', 'coca', 'siglip', 'eva02'],
    )
    parser.add_argument(
        "--finetuning",
        type=str,
        default='zs',
        choices=['adapter', 'attention', 'bias', 'lora', 'full', 'zs']
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default='metric_scores.csv'
    )
    parser.add_argument(
        "--root",
        type=str,
    )
    parser.add_argument(
        "--prop",
        type=float,
        default=1
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=['gap', 'uniformity_image', 'uniformity_text', 'inter_images', 'inter_text', 'IIMM']
    )
    
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    pprint_args(args)
    df = pd.DataFrame()
    metrics_dict = {
                    'gap': gap,
                    'uniformity_image': uniformity_image,
                    'uniformity_text': uniformity_text,
                    'inter_images': inter,
                    'inter_text': inter,
                    'IIMM': IIMM
                    }
    print('Getting metric scores')
    selected_metrics = {k:v for k,v in metrics_dict.items() if k in args.metrics}
    get_metric_scores(datasets=args.datasets, model=args.model, finetuning=args.finetuning, root=args.root, metrics=selected_metrics, outpath=args.outpath, prop=args.prop)
    print('complete')
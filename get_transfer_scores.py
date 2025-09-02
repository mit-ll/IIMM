import os
import time
from collections import defaultdict
import argparse

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import normalize

from utils import *
from models.wrappers import OpenCLIP, SigLip, CoCa
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize
from metrics import IIMM
from transferability_metrics import (
    SFDA_Score,
    NCTI_Score,
    Transrate,
    H_Score,
    EMMS,
    GBC,
    emms_clip_label_embeddings,
    emms_bert_label_embeddings,
    emms_gpt2_label_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['Cars','CIFAR100','DTD','EuroSAT','GTSRB','MNIST','RESISC45','STL10','SUN397','SVHN'],
        help='Dataset to evaluate',
        required=True,
    )
    parser.add_argument(
        '--eval-models',
        type=str,
        nargs='+',
        choices=['clip', 'coca', 'eva02', 'siglip'],
        help='Options for models to evaluate. Can pass multiple as space-separated list'
    )
    parser.add_argument(
        '--metrics',
        type=str, 
        nargs='+', 
        choices=['iimm', 'sfda', 'ncti', 'transrate', 'emms', 'hscore', 'gbc_spherical', 'gbc_diagonal'],
        help='Transferability metric to compute. Can pass multiple as space-separated list'
    )
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--val-p', type=float, default=0.1, help='Percentage of the training data to use as validation data during training')
    parser.add_argument('--cache-dir', type=str, default='./cache', help='Directory in which to save feature and label numpy matrices')
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--progress', action='store_true', help='Show progress bar for feature extractions')

    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    pprint_args(args)
    
    score_dict = defaultdict(dict)
    exec_time_dict = defaultdict(dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for model_name in args.eval_models:
        # Load model and transforms
        print(f'Loading {model_name}...')
        model, preprocess = create_model_and_transforms(model_name)
        if model_name == 'siglip':
            model = SigLip(model=model)
        elif model_name == 'coca':
            model = CoCa(coca_model=model)
        else:
            model = OpenCLIP(model)
        model.to(device=device)

        # Load data
        print(f'Loading {args.dataset}...')
        dataset_name = args.dataset+'Val'
        data = get_dataset(dataset_name=dataset_name,
                        preprocess=preprocess, 
                        batch_size=args.bs, 
                        num_workers=args.num_workers, 
                        val_fraction=args.val_p, 
                        max_val_samples=5000)

        im_features_npy_path = os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_im_features.npy')
        txt_features_npy_path = os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_text_features.npy')
        labels_npy_path = os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_labels.npy')
        img_embed_duration = txt_embed_duration = 0
        try:
            X_features, txt_features, y_labels = np.load(im_features_npy_path), np.load(txt_features_npy_path), np.load(labels_npy_path)
            print(f'Found cached {model_name} features and labels for {args.dataset}')
        except FileNotFoundError:
            # Gather target data features and labels
            print(f'No cached {model_name} features or labels found for {args.dataset}. Extracting features and labels...')
            X_features = []
            txt_features = []
            y_labels = []
            train_loader = tqdm(data.train_loader) if args.progress else data.train_loader
            model.eval()

            # Get text embeddings
            tokenizer = get_tokenizer(model_name)
            tokenized_text = get_tokenized_captions(args.dataset, data, tokenizer).to(device)
            txt_embed_start_time = time.time()
            txt_features = model.encode_text(tokenized_text).detach().cpu().numpy()
            txt_embed_end_time = time.time()
            txt_embed_duration = txt_embed_end_time - txt_embed_start_time

            # Get image embeddings and labels
            img_embed_start_time = time.time()
            with torch.no_grad():
                for batch in train_loader:
                    batch = maybe_dictionarize(batch)
                    x = batch['images'].to(device)
                    y = batch['labels']

                    y_labels.append(y.numpy())
                    img_f = model.encode_image(x).detach().cpu().numpy()
                    X_features.append(img_f)
            img_embed_end_time = time.time()
            img_embed_duration = img_embed_end_time - img_embed_start_time
            
            # Normalize and combine
            X_features = normalize(np.concatenate([x for x in X_features]), axis=1)
            txt_features = normalize(txt_features, axis=1)
            y_labels = np.concatenate([y for y in y_labels])
            
            # Save features and labels
            if not os.path.exists(args.cache_dir):
                os.makedirs(args.cache_dir)
            np.save(im_features_npy_path, X_features)
            np.save(txt_features_npy_path, txt_features)
            np.save(labels_npy_path, y_labels)

        for metric in args.metrics:
            calc_start_time = time.time()
            if metric == 'sfda':
                score_dict[metric][model_name] = SFDA_Score(X_features, y_labels)
            elif metric == 'ncti':
                score_dict[metric][model_name] = NCTI_Score(X_features, y_labels)
            elif metric == 'transrate':
                score_dict[metric][model_name] = Transrate(X_features, y_labels)
            elif metric == 'hscore':
                score_dict[metric][model_name] = H_Score(X_features, y_labels)
            elif metric == 'gbc_spherical':
                score_dict[metric][model_name] = GBC(X_features, y_labels, gaussian_type='spherical')
            elif metric == 'gbc_diagonal':
                score_dict[metric][model_name] = GBC(X_features, y_labels, gaussian_type='diagonal')
            elif metric == 'iimm':
                score_dict[metric][model_name] = IIMM(X_features, txt_features, y_labels)
            elif metric == 'emms':
                # Extract F-labels of target data using CLIP, GPT-2, and BERT
                try:
                    label_embeds_clip = np.load(os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_emms_label_embeds_clip.npy'))
                    label_embeds_gpt2 = np.load(os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_emms_label_embeds_gpt2.npy'))
                    label_embeds_bert = np.load(os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_emms_label_embeds_bert.npy'))
                except FileNotFoundError:
                    label_embeds_clip = emms_clip_label_embeddings(data.classnames, load_path="huggingface/CLIP-ViT-g-14-laion2B-s12B-b42K", device=device)
                    label_embeds_gpt2 = emms_gpt2_label_embeddings(data.classnames, load_path="huggingface/gpt2-medium", device=device)
                    label_embeds_bert = emms_bert_label_embeddings(data.classnames, load_path="huggingface/bert-large-uncased", device=device)
                    np.save(os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_emms_label_embeds_clip.npy'), label_embeds_clip)
                    np.save(os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_emms_label_embeds_gpt2.npy'), label_embeds_gpt2)
                    np.save(os.path.join(args.cache_dir, f'{args.dataset}_{model_name}_emms_label_embeds_bert.npy'), label_embeds_bert)

                y_labels_clip = np.zeros([y_labels.shape[0], label_embeds_clip.shape[1]])
                y_labels_gpt2 = np.zeros([y_labels.shape[0], label_embeds_gpt2.shape[1]])
                y_labels_bert = np.zeros([y_labels.shape[0], label_embeds_bert.shape[1]])
                for i in range(y_labels.shape[0]):
                    y_labels_clip[i] = label_embeds_clip[y_labels[i]]
                    y_labels_gpt2[i] = label_embeds_gpt2[y_labels[i]]
                    y_labels_bert[i] = label_embeds_bert[y_labels[i]]
                y_labels_emms = np.stack((y_labels_clip, y_labels_gpt2, y_labels_bert), axis=2)

                score_dict[metric][model_name] = EMMS(X_features, y_labels_emms)
            else:
                raise NotImplementedError
            
            calc_end_time = time.time()
            calc_duration = calc_end_time - calc_start_time
            
            # Record computational costs in terms of time
            embedding_duration = img_embed_duration if metric != 'iimm' else img_embed_duration + txt_embed_duration
            exec_time_dict[metric][model_name] = {
                'embedding_time': embedding_duration,
                'calculation_time': calc_duration,
                'total_time': embedding_duration + calc_duration
            }
        print('\n')

    # Normalize and consolidate NCTI scores
    if 'ncti' in args.metrics:
        seli_scores = [score_dict['ncti'][m][0] for m in args.eval_models]
        ncc_scores = [score_dict['ncti'][m][1] for m in args.eval_models]
        vc_scores = [score_dict['ncti'][m][2] for m in args.eval_models]

        if len(args.eval_models) > 1:
            for model in args.eval_models:
                ncti_extra_start = time.time()

                normal_seli = (score_dict['ncti'][model][0] - min(seli_scores))/(max(seli_scores) - min(seli_scores))
                normal_ncc = (score_dict['ncti'][model][1] - min(ncc_scores))/(max(ncc_scores) - min(ncc_scores))
                normal_vc = (score_dict['ncti'][model][2] - min(vc_scores))/(max(vc_scores) - min(vc_scores))
                score_dict['ncti'][model] = normal_ncc + normal_seli - normal_vc
                
                # Add extra time taken for normalization to ncti calculation time
                ncti_extra_duration = time.time() - ncti_extra_start
                exec_time_dict['ncti'][model]['calculation'] += ncti_extra_duration
                exec_time_dict['ncti'][model]['total'] += ncti_extra_duration
        else:
            score_dict['ncti'][model_name] = ncc_scores[0] + seli_scores[0] - vc_scores[0]
        
    print('Raw score results:')
    raw_df = pd.DataFrame.from_dict(score_dict, orient='index')
    print(raw_df)

    print('\nExecution times:')
    exec_df = pd.DataFrame.from_dict({(metric, model_name): times 
                             for metric, model_dict in exec_time_dict.items() 
                             for model_name, times in model_dict.items()},
                            orient='index')
    exec_df.index = pd.MultiIndex.from_tuples(exec_df.index, names=["metric", "model"])
    print(exec_df)

    # Save results
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Adjust raw scores format for csv
    raw_df = raw_df.reset_index().rename(columns={'index':'metric'})
    raw_df = pd.melt(raw_df, id_vars=['metric'], value_vars=args.eval_models, var_name='model', value_name='score')
    raw_df['train_data'] = pd.Series([args.dataset]*len(raw_df))
    raw_df = raw_df.reindex(columns=['train_data','model','metric','score'])
    raw_df.to_csv(os.path.join(args.results_dir, f'{args.dataset}_transfer_scores.csv'), index=False)

if __name__ == '__main__':
    main()
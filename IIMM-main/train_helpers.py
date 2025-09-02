import os
import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds import get_dataset as get_wilds_dataset
from wilds.common.data_loaders import get_eval_loader as get_wilds_eval_loader

from utils import get_tokenized_captions
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize


def train(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    model_save_path: str,
    tokenized_text: torch.Tensor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    nepochs: int,
    progress: bool=False,
):

    loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0

    dataset = "fmow" if isinstance(train_loader.dataset, WILDSSubset) else "other"
    n = len(train_loader.dataset)

    tokenized_text = tokenized_text.to(device=device)
    
    for epoch in range(nepochs):
        model.train()
        # Wrap train_loader
        train_loader = tqdm(train_loader) if progress else train_loader
        
        train_loss = 0.0
        train_acc = 0.0
        t_start = datetime.datetime.now()
        
        # Begin training
        for batch in train_loader:
            if dataset == "fmow":
                x, y, _ = batch
            else:
                batch = maybe_dictionarize(batch)
                x = batch['images']
                y = batch['labels']
            x, y = x.to(device=device), y.to(device=device)
            
            # Get logits
            with torch.amp.autocast('cuda'):
                outputs, _ = model(x, tokenized_text)
                loss = loss_fn(outputs, y)
            
            y_pred = outputs.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_true = y.cpu().numpy()
            train_acc += np.sum(y_pred == y_true)

            # Compute gradient and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        t_end = datetime.datetime.now()
        loss = train_loss / len(train_loader)
        acc = (train_acc / n) * 100
            
        # Validation
        val_acc = model_eval(model, device, val_loader, tokenized_text, progress)
        print(f"Val accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), model_save_path)
            best_val_acc = val_acc

        print(f"Epoch {epoch+1:>3d}; Time: {t_end - t_start}; Loss = {loss:.4f}: acc = {acc:.4f}")

    print("Training complete, model saved:", model_save_path)
    return model, best_val_acc


def model_eval(
        model: nn.Module,
        device: torch.device,
        loader: torch.utils.data.DataLoader,
        tokenized_text: torch.Tensor,
        progress: bool=False,
):  
    model.eval()

    acc = 0
    
    dataset = "fmow" if isinstance(loader.dataset, WILDSSubset) else "other"
        
    with torch.no_grad():
        
        tokenized_text = tokenized_text.to(device=device)

        n = len(loader.dataset)
        loader = tqdm(loader) if progress else loader
        
        for batch in loader:
            
            if dataset == "fmow":
                x, y, _ = batch
            else:
                batch = maybe_dictionarize(batch)
                x = batch['images']
                y = batch['labels']
            x = x.to(device=device)
            
            # Get logits
            outputs, _ = model(x, tokenized_text) # scaled cosine similarity

            y_pred = outputs.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_true = y.numpy()
            acc += np.sum(y_pred == y_true)

        results = (acc / n) * 100

    return results

def get_eval_loader(
         dataset: str,
         num_workers: int,
         batch_size: int,
         preprocess: transforms.Compose,
         tokenizer,
):
    if 'fmow' in dataset.lower():
        
        fmow = get_wilds_dataset(dataset="fmow", download=False, root_dir=os.path.expanduser("~/data"))
        if 'ood' in dataset:
            # out of distribution test set
            data = fmow.get_subset(
                "test",
                transform=preprocess,
            )
        else:
            # in distribution test set
            data = fmow.get_subset(
                "id_test",
                transform=preprocess,
            )
        test_loader = get_wilds_eval_loader(
            "standard",
            dataset=data,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=batch_size,
        )
            
    else:
        data = get_dataset(dataset_name=dataset,
                    preprocess=preprocess, 
                    batch_size=batch_size, 
                    num_workers=num_workers)
        
        test_loader = data.test_loader
        
    tokenized_captions = get_tokenized_captions(dataset, data, tokenizer)
    
    return test_loader, tokenized_captions
    
    
def embedding_angles(
        model: nn.Module,
        ft_model: nn.Module,
        device: torch.device,
        loader: torch.utils.data.DataLoader,
        tokenized_text: torch.Tensor,
        data
):  
    # Load finetuned model weights
    
    ft_model.eval()
    model.eval()
    
    print('Getting embedding angles diffs')
    angle_diffs = np.array([], dtype=np.int64).reshape(0,len(tokenized_text)+1)   
    acc = 0
 
    with torch.no_grad():
        for batch in loader:
            
            if "fmow" in data:
                x, y, _ = batch
            else:
                batch = maybe_dictionarize(batch)
                x = batch['images']
                y = batch['labels']
            x = x.to(device=device)
            
            # Get logits
            pt_outputs, _ = model(x, tokenized_text) # scaled cosine similarity
            ft_outputs, _ = ft_model(x, tokenized_text) # scaled cosine similarity
            
            # get accuracy
            ft_outputs = ft_outputs.detach().cpu().numpy()
            y_pred = np.argmax(ft_outputs, axis=1)
            y_true = y.numpy()
            acc += np.sum(y_pred == y_true)


            # get cosine similarity 
            pt_outputs = pt_outputs.detach().cpu().numpy() / 100
            ft_outputs = ft_outputs / 100
            # get angle diffs
            y_true = y.numpy().reshape((len(y), 1))
            _angle_diffs = np.append(np.abs(np.arccos(pt_outputs) - np.arccos(ft_outputs)), y_true, 1)
            # add angle diffs of batch to array
            angle_diffs = np.concatenate([angle_diffs, _angle_diffs])
            
        results = acc / len(loader.dataset)
        print("model accuracy ", results)
    return angle_diffs
    
   
    

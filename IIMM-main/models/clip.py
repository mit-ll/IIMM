import torch
import torch.nn as nn
import open_clip
import numpy as np
import re

from .common import Adapter, LoraMLP, LoraMultiheadAttention
    

class CLIPAdapter(nn.Module):
    def __init__(self, clip_model, reduction=4):
        super().__init__()
        # freeze gradient of all layers in base model
        for param in clip_model.parameters():
            param.requires_grad = False
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.base_encode_image = clip_model.encode_image
        self.base_encode_text = clip_model.encode_text
        self.image_adapter = Adapter(c_in=512, reduction=reduction)
        self.text_adapter = Adapter(c_in=512, reduction=reduction)
        
    def to(self, device):
        self.clip_model.to(device=device)
        self.image_adapter.to(device=device)
        self.text_adapter.to(device=device)
        
    def forward(self, images, tokenized_text):
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokenized_text)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        probs = logits.softmax(dim=-1)
        return logits, probs
    
    def encode_image(self, images):
        # Get embeddings of base model
        image_embeddings = self.base_encode_image(images)
        
        # get embeddings of adapters
        x = self.image_adapter(image_embeddings)
        
        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_embeddings
        
        # normalize feature vectors
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, tokenized_text):
        # Get embeddings of base model
        text_embeddings = self.base_encode_text(tokenized_text) 
        
        # get embeddings of adapters
        y = self.text_adapter(text_embeddings)
        
        ratio = 0.2
        text_features = ratio * y + (1 - ratio) * text_embeddings
        
        # normalize feature vectors
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def state_dict(self):
        sd = {}
        sd['image_adapter'] = self.image_adapter.state_dict()
        sd['text_adapter'] = self.text_adapter.state_dict()
        return sd
        
    def load_state_dict(self, params, strict=True):
        self.image_adapter.load_state_dict(params['image_adapter'], strict=strict)
        self.text_adapter.load_state_dict(params['text_adapter'], strict=strict)


class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model, n_classes, num_ftrs=512):
        super().__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(num_ftrs, n_classes)

        # Leave trainable parameters only in linear layer
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x, *args):
        img_f = self.clip_model.encode_image(x)
        img_f = img_f.float()
        logits = self.fc(img_f)
        probs = logits.softmax(dim=-1)
        return logits, probs
    

class CustomCLIP():
    def __init__(self, clip_model, model_path=None):
        self.num_named_params = 302
        self.model = clip_model
        if model_path:
            self.params = torch.load(model_path)
            try:
                self.params['positional_embedding']
            except KeyError:
                new_params = {}
                for k, v in self.params.items():
                    new_params[k.split('clip_model.')[1]] = v
                self.params = new_params
            self.model.load_state_dict(self.params, strict=True)
                
    def set_specified_train_layers(self, train_blocks=12, all_layers=False, reg='attention'):
        '''
        train_layers: list[int]
        '''
        layer_names = list(self.model.state_dict().keys())
        # freeze parameters other than specified layers
        blocks = [f'resblocks.{x}.' for x in np.arange(train_blocks)]
        if not all_layers:
            if reg == 'attention':
                regex = re.compile(r'resblocks.*attn.in.*weight')
            else:
                regex = re.compile(r'resblocks.*bias')
            layer_names = [l for l in layer_names if regex.search(l)]
            train_layers = [l for l in layer_names if np.sum([b in l for b in blocks])]
        for l in train_layers:
            print(l)
        for name, param in self.model.named_parameters():
            if name not in train_layers:
                param.requires_grad = False


class LoRACLIP(open_clip.model.CLIP):
    def __init__(self, clip_model, include_mlp=False, rank=4):
        super(open_clip.model.CLIP, self).__init__()
        
        self.clip_model = clip_model
        self.rank = rank
        self.include_mlp = include_mlp

        # Replace visual transformer layers with lora
        for resblock in self.clip_model.visual.transformer.resblocks:
            resblock.attn = LoraMultiheadAttention(resblock.attn, rank=self.rank)
            if include_mlp:
                resblock.mlp.c_fc = LoraMLP(resblock.mlp.c_fc, rank=self.rank)

        # Replace language transformer layers with lora
        for resblock in self.clip_model.transformer.resblocks:
            resblock.attn = LoraMultiheadAttention(resblock.attn, rank=self.rank)
            if include_mlp:
                resblock.mlp.c_fc = LoraMLP(resblock.mlp.c_fc, rank=self.rank)
        
        # Freeze all layer weights aside from LoRA
        for name, param in self.clip_model.named_parameters():
            if "lora" not in name and param.requires_grad:
                param.requires_grad = False

    def forward(self, x, text):
        img_f, txt_f, scale = self.clip_model(x, text)
        logits = scale * img_f @ txt_f.T
        probs = logits.softmax(dim=-1)
        return logits, probs
    
    def load_state_dict(self, params, strict=True):
        self.clip_model.load_state_dict(params, strict=strict)
        
    def encode_image(self, images):
        return self.clip_model.encode_image(images)
    
    def encode_text(self, tokenized_text):
        return self.clip_model.encode_text(tokenized_text)
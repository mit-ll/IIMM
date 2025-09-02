import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, tokenized_text):
        img_f, txt_f, scale = self.clip_model(x, tokenized_text)
        logits = scale * img_f @ txt_f.T
        probs = logits.softmax(dim=-1)
        return logits, probs
    
    def encode_image(self, images):
        return self.clip_model.encode_image(images)
    
    def encode_text(self, tokenized_text):
        return self.clip_model.encode_text(tokenized_text)
    
    def load_state_dict(self, params, strict=True):
        self.clip_model.load_state_dict(params, strict=strict)
        
        
class CoCa(nn.Module):
    # For image classification using CoCa vision and text encoders

    def __init__(self, coca_model):
        super().__init__()
        self.coca_model = coca_model

    def forward(self, x, tokenized_text):
        img_f = self.coca_model.encode_image(x)
        txt_f = self.coca_model.encode_text(tokenized_text)
        logits = img_f @ txt_f.T * self.coca_model.logit_scale.exp()
        if self.coca_model.logit_bias:
            logits += self.coca_model.logit_bias
        probs = logits.softmax(dim=-1)

        return logits, probs

    def encode_image(self, image):
        return self.coca_model.encode_image(image)
    
    def encode_text(self, text):
        return self.coca_model.encode_text(text)
    
    def load_state_dict(self, params, strict=True):
        self.coca_model.load_state_dict(params, strict=strict)
        

class SigLip(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, tokenized_text):
        img_features = self.encode_image(x)
        text_features = self.encode_text(tokenized_text)
        img_features = F.normalize(img_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = img_features @ text_features.T * self.model.logit_scale.exp() + self.model.logit_bias
        probs = torch.sigmoid(logits)
        
        return logits, probs
    
    def encode_image(self, images):
        return self.model.encode_image(images)
    
    def encode_text(self, tokenized_text):
        return self.model.encode_text(tokenized_text)
    
    def load_state_dict(self, params, strict=True):
        self.model.load_state_dict(params, strict=strict)
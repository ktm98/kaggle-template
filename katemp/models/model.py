import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ImageModel(nn.Module):
    def __init__(self, model_name, pretrained=False, target_size=1, *args, **kwargs):
        super().__init__()
        # self.cfg = cfg
        self.model = timm.create_model(model_name, pretrained=pretrained, *args, **kwargs)
        if 'efficientnet' in model_name:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, target_size)
        elif 'resnet' in model_name:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, target_size)
        elif 'nfnet' in model_name:
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(self.n_features, target_size)
        elif ('vit' in model_name) or ('swin' in model_name):
            self.n_features = self.model.head.in_features
            self.model.head = nn.Linear(self.n_features, target_size)


        # self.n_features = self.model.head.fc.in_features
        # self.model.head.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def forward(self, x):
        output = self.model(x)
        return output
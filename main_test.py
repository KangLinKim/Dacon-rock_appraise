import torch
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import glob
import os
from PIL import Image
from collections import defaultdict, Counter
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from sklearn.manifold import TSNE
import json
import math
import matplotlib.pyplot as plt


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

print(torch.__version__)


class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epoch = 60

        self.image_size = (224, 224)
        self.lr = 0.00005
        self.weight_decay = 0.001
    
        self.train_data_dir = r'C:\Users\user\Downloads\open (1)\train'
        self.test_data_dir = r'C:\Users\user\Downloads\open (1)\test'
        self.test_size = 0.2
        self.batch_size = 16
        self.sample_num = 8000

        self.visualize_epoch = 5
        self.visualize_save_dir = os.path.join(os.getcwd(), 'visualized_features')

config = Config()
print(config.device)




class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j, z_label=None):
        batch_size = z_i.size(0)
        
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]

        labels = z_label.repeat_interleave(2)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(z_i.device)
        mask.fill_diagonal_(0)

        logits_mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        exp_sim = torch.exp(sim_matrix) * logits_mask

        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        log_prob = log_prob * logits_mask

        loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        # loss = -loss.mean()

        return loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.view(-1, 1)

        log_p = log_probs.gather(1, targets).squeeze(1)
        p_t = probs.gather(1, targets).squeeze(1)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = self.alpha[targets.squeeze()]
            else:
                alpha_t = self.alpha
            loss = -alpha_t * (1 - p_t) ** self.gamma * log_p
        else:
            loss = -(1 - p_t) ** self.gamma * log_p

        return loss
    

class custom_model(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()

        if isinstance(backbone, str):
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        elif isinstance(backbone, nn.Module):
            self.backbone = backbone
        else:
            raise

        with torch.no_grad():
            dummy = torch.randn(1, 3, config.image_size[0], config.image_size[1])
            num_features = self.backbone(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)

        if return_features:
            return self.head(features), features
        
        return self.head(features)
    
    def forward_features(self, x):
        return self.backbone(x)
    


if __name__ == '__main__':
    item_list = glob.glob(rf"{config.train_data_dir}\*\*")
    len(item_list)
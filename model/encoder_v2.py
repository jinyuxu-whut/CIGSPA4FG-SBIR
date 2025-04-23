import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from torch.nn import functional as F

# v2
class Encoder(nn.Module):
    def __init__(self, feature_type="mid", k=0.7):
        super(Encoder, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_type=feature_type
        if feature_type == "high":
            self.backbone = nn.Sequential(
                OrderedDict(list(resnet50.named_children())[:-2])
            )
            self.fea_wh = 8
            self.fea_channel = 2048
        elif feature_type == "mid":
            self.backbone = nn.Sequential(
                OrderedDict(list(resnet50.named_children())[:-3])
            )
            self.fea_wh = 16
            self.fea_channel = 1024

            self.fc_attn = nn.Linear(self.fea_channel, 1)
            self.global_avg = nn.AdaptiveAvgPool1d(1)

            # print("******new Net v2************")
        elif feature_type == "global":
            self.backbone = nn.Sequential(
                OrderedDict(list(resnet50.named_children())[:-2])
            )
            self.fea_wh = 1
            self.fea_channel = 2048
        
        self.k = k
        # print("******using filter************ k:" + str(self.k))
    def forward(self, x):
        fea = self.backbone(x)
        if self.feature_type == "global":
            fea = torch.nn.functional.adaptive_avg_pool2d(fea, 1)
            fea = torch.flatten(fea, 1)
            out = torch.nn.functional.normalize(fea, dim=1)
        elif self.feature_type == "mid":
            fea = torch.nn.functional.normalize(fea, dim=1)
            out = fea.flatten(2).permute(0, 2, 1)
            # + filter
            attn0 = out @ out.transpose(-2, -1)
            # b, n, n
            q1 = self.global_avg(attn0).transpose(-2, -1)   #  patch-wise mean similarity of patches
            # b, 1, n
            attn1 = self.fc_attn(out)   
            # b, n, 1
            attn1 = attn1 @ q1
            attn0 = attn0 + attn1
            # b, n, n
            b, n, f = out.shape
            keep_num = int(n * self.k)
            keep_index = torch.topk(torch.sum(attn0, -1), keep_num, -1, largest=False)[1]
            keep_index,_=torch.sort(keep_index,dim=1)
            out = torch.gather(
                out, 1, keep_index.unsqueeze(2).expand(b, keep_num, f)
            )
            # return out
        elif self.feature_type == "high":
            fea = torch.nn.functional.normalize(fea, dim=1)
            out = fea.flatten(2).permute(0, 2, 1)
        return out
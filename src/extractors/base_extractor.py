import torch
from typing import List
from .transparent import build_transparent


class FeatureExtractor():
    def __init__(self, model_name: str, extract_blocks: List[int],
                 pooling='avg', device: str = 'cuda'):
        self.net = build_transparent(model_name=model_name, pretrained=True,
                                     extract_blocks=extract_blocks, freeze=True)
        self.net = self.net.to(device=device)
        self.extract_blocks = extract_blocks
        self.pooling = pooling

    def __call__(self, x: torch.Tensor):
        feature_list = self.net(x)
        feats = []
        for feat in feature_list:
            if self.pooling == 'min':
                feats.append(torch.amin(feat, dim=(2, 3)))
            elif self.pooling == 'max':
                feats.append(torch.amax(feat, dim=(2, 3)))
            elif self.pooling == 'avg':
                feats.append(torch.mean(feat, dim=(2, 3)))
            elif self.pooling == 'minmax':
                min = torch.amin(feat, dim=(2, 3))
                max = torch.amax(feat, dim=(2, 3))
                x = torch.cat([min, max], axis=1)
                feats.append(x)
            elif self.pooling == 'minavg':
                min = torch.amin(feat, dim=(2, 3))
                avg = torch.mean(feat, dim=(2, 3))
                x = torch.cat([min, avg], axis=1)
                feats.append(x)
            elif self.pooling == 'maxavg':
                max = torch.amax(feat, dim=(2, 3))
                avg = torch.mean(feat, dim=(2, 3))
                x = torch.cat([max, avg], axis=1)
                feats.append(x)
            elif self.pooling == 'minmaxavg':
                min = torch.amin(feat, dim=(2, 3))
                max = torch.amax(feat, dim=(2, 3))
                avg = torch.mean(feat, dim=(2, 3))
                x = torch.cat([min, max, avg], axis=1)
                feats.append(x)
            else:
                raise TypeError("pooling type expected to be 'max' or 'avg' or 'maxavg'")

        feats = torch.cat(feats, dim=1)

        return feats

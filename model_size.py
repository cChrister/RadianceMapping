import torch
from torchvision import models


def model_size(checkpoint):
    params = 0
    size_all_mb = 0
    for key, value in checkpoint.items():
        model = checkpoint[key]
        params += model.numel()
        size_all_mb += (model.numel() * model.element_size())
    #print(params)
    return (size_all_mb / (1024 ** 2))

print('bpcr model size: {:.3f}MB'.format(model_size(torch.load("vanilla_bpcr.pkl"))))
print('bpcr_mpn model size: {:.3f}MB'.format(model_size(torch.load("bpcr_mpn.pkl"))))
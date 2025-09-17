import torch
from torch import nn
import torchvision
from torchvision.models import ConvNeXt_Base_Weights
import torch.nn.functional as F

# This ConvNeXt based encoder class is adapted from the codebase of the original study (Ramos et al., 2024).
# Link to their GitHub repository: https://github.com/Leo-Thomas/ConvNeXt-for-Image-Captioning/tree/main
# The original study (Ramos et al., 2024) seem to have adapted their code from another repository (Vinodababu, 2019) 
# which is a popular open source implementation of the 'Show, Attend and Tell' paper (Xu et al., 2015).
# Link to the (Vinodababu, 2019) repository: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=7):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        convnext = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.convnext = convnext.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        out = self.convnext(images)  # (batch_size, 768, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 768, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 768)
        return out
    
    def fine_tune(self, fine_tune=True, startingLayer=7):   # A starting layer parameter is added to allow fine-tuning
        for p in self.convnext.parameters():                # from specific layers in this stidy
            p.requires_grad = False
        for c in list(self.convnext.children())[startingLayer:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


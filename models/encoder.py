import torch
from torch import nn
import torchvision
from torchvision.models import ConvNeXt_Base_Weights
import torch.nn.functional as F


device = torch.device("cuda")

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=7):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size  #Be aware about encode dim
        #convnext = torchvision.models.convnext_tiny(pretrained=True) 
        # convnext = torchvision.models.convnext_small(pretrained=True) 
        # convnext = torchvision.models.convnext_base(pretrained=True) 
        #convnext = torchvision.models.convnext_large(pretrained=True) 

        convnext = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(convnext.children())[:-2]
        self.convnext = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        out = self.convnext(images)  # (batch_size, 768, image_size/32, image_size/32)
        _, _, h, w = out.shape
        pad_h = (self.enc_image_size - h % self.enc_image_size) % self.enc_image_size
        pad_w = (self.enc_image_size - w % self.enc_image_size) % self.enc_image_size
        # Apply padding
        out = F.pad(out, (0, pad_w, 0, pad_h))  # Pad the height and width
        out = self.adaptive_pool(out)  # (batch_size, 768, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 768)
        return out
    
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.convnext.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.convnext.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


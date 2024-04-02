import segmentation_models_pytorch as smp
import torch.nn as nn


# check other pretrained models and give possibility to choose one
# write own Unet - just choose the best pretrained encoder

class UNet:
    def __init__(self, device):
        self.encoder = "resnet18"
        self.num_classes = 1
        self.device = device

        self.model = smp.Unet(encoder_name=self.encoder, encoder_weights="imagenet", classes=self.num_classes,
                              activation="sigmoid")

        self.model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=False)
        self.model = self.model.to(self.device)

        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, "imagenet")

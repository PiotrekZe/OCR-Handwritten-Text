import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms


# model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", classes=1,
#                  activation="sigmoid")
#
# model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
#                                 padding=(3, 3), bias=False)
#
# model_path = "E:/AI/Handwritten Text Recognition/temp_outputs/model_30.pt"
# model.load_state_dict(torch.load(model_path))
#
# transform = transforms.Compose(
#     [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
#
# size = [224, 224]
#
# image = Image.open("D:/Databases/06/forms_obciete/a01-000u.png").convert("RGB")
# plt.imshow(image)
# plt.show()
# # image = Image.eval(image, lambda x: 255 - x)
# image = image.resize((size[1], size[0]), resample=Image.BILINEAR)
# plt.imshow(image)
# plt.show()
#
# image = transform(image)
# image_with_batch = image.unsqueeze(0).to('cpu')
# criterion = torch.nn.BCEWithLogitsLoss()
# print(image_with_batch.shape)
# model = model.to("cpu")
# model.eval()
#
# output = model(image_with_batch)
# plt.imshow(output[0].detach().numpy().squeeze())
# plt.show()


def segment_image(path_to_model, img):
    # define model with loss function
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", classes=1,
                     activation="sigmoid")
    model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                    padding=(3, 3), bias=False)

    criterion = torch.nn.BCEWithLogitsLoss()

    # load pretrained weights and set model to evaluation
    model_path = "E:/AI/Handwritten Text Recognition/temp_outputs/model_30.pt"
    model.load_state_dict(torch.load(model_path))
    model = model.to("cpu")
    model.eval()

    # image options
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    size = [224, 224]

    # here i should make sure that image is converted into black and white (text should be in white)

    # read image
    image = Image.open("D:/Databases/06/forms_obciete/a01-000u.png").convert("RGB")
    image = image.resize((size[1], size[0]), resample=Image.BILINEAR)
    image = transform(image)
    image = image.unsqueeze(0).to('cpu')

    output = model(image)
    plt.imshow(output[0].detach().numpy().squeeze())
    plt.show()
    return output


segment_image("tml", "tml")

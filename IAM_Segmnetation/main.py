import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import CustomDataset
import Dataset
import Model
import RunModel
import segmentation_models_pytorch as smp
import torch.nn as nn


def main():
    images_path = "D:/Databases/06/forms_obciete"
    masks_path = "D:/Databases/06/forms_masks1"
    batch_size = 8
    learning_rate = 0.0001
    epochs = 100
    device = "cuda"

    dataset = Dataset.Dataset(images_path, masks_path)
    train_images, test_images, train_masks, test_masks = dataset.read_dataset()

    train_dataset = CustomDataset.CustomDataset(images=train_images, masks=train_masks, size=[224, 224])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = CustomDataset.CustomDataset(images=test_images, masks=test_masks, size=[224, 224])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # model = Model.UNet(device)
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", classes=1,
                     activation="sigmoid")

    model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                    padding=(3, 3), bias=False)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    run_model = RunModel.RunModel(epochs, device, train_loader, test_loader)

    train_loss_tab, test_loss_tab = [], []
    p = "E:/AI/Handwritten Text Recognition/temp_outputs/"
    for epoch in range(epochs):
        train_loss, train_mask, train_output, train_image = run_model.train_model(model, optimizer, criterion)
        test_loss, test_mask, test_output, test_image = run_model.test_model(model, criterion)
        train_loss_tab.append(train_loss)
        test_loss_tab.append(test_loss)

        print(f"Epoch {epoch + 1}, train loss: {train_loss}, test loss: {test_loss}")
        plt.imsave(p + "train_mask_" + str(epoch + 1) + ".jpg", train_mask.squeeze().cpu().detach().numpy())
        plt.imsave(p + "test_mask_" + str(epoch + 1) + ".jpg", test_mask.squeeze().cpu().detach().numpy())
        plt.imsave(p + "train_output_" + str(epoch + 1) + ".jpg", train_output.squeeze().cpu().detach().numpy())
        plt.imsave(p + "test_output_" + str(epoch + 1) + ".jpg", test_output.squeeze().cpu().detach().numpy())
        plt.imsave(p + "train_image_" + str(epoch + 1) + ".jpg", train_image.squeeze().cpu().detach().numpy())
        plt.imsave(p + "test_image_" + str(epoch + 1) + ".jpg", test_image.squeeze().cpu().detach().numpy())

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), p + "model_" + str(epoch + 1) + ".pt")



if __name__ == '__main__':
    main()

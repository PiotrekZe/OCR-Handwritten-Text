import torch
from torch.utils.data import DataLoader

import CustomDataset
import Dataset
import RunModel
import Decoder
import Model
import utils


def main():
    config_data = utils.read_config("config_file.json")

    learning_rate = config_data['model']['learning_rate']
    batch_size = config_data['model']['batch_size']
    epochs = config_data['model']['epochs']
    device = config_data['model']['device']
    num_layers = config_data['model']['num_layers']
    dims = config_data['model']['dims']

    path = config_data['file']['path']
    height = config_data['file']['height']
    width = config_data['file']['width']
    path_to_save = config_data['file']['path_to_save']
    num_channels = config_data['file']['num_channels']

    size = (height, width)
    dataset = Dataset.Dataset(path)
    (train_paths, test_paths, train_targets, test_targets, train_targets_lengths, test_targets_lengths,
     train_original_targets, test_original_targets, target_classes) = dataset.read_dataset()

    train_dataset = CustomDataset.CustomDataset(paths=train_paths, targets=train_targets,
                                                original_targets=train_original_targets, size=size,
                                                num_channels=num_channels, target_lengths=train_targets_lengths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = CustomDataset.CustomDataset(paths=test_paths, targets=test_targets,
                                               original_targets=test_original_targets, size=size,
                                               num_channels=num_channels, target_lengths=test_targets_lengths)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    crnn = Model.CRNN(size=size, num_chars=len(target_classes), num_channels=num_channels, device=device, dims=dims,
                      num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=learning_rate)  # 3e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)

    decoder = Decoder.Decoder(0)
    run_model = RunModel.RunModel(epochs, device, train_loader, test_loader, decoder, target_classes)

    for epoch in range(epochs):
        (train_decoded_outputs, train_running_loss, train_original_targets_list,
         train_accuracy, train_cer) = run_model.train_model(crnn, optimizer)
        (test_decoded_outputs, test_running_loss, test_original_targets_list,
         test_accuracy, test_cer) = run_model.test_model(crnn)

        print(f"Epoch: {epoch}, Train loss: {train_running_loss}, Test loss: {test_running_loss}")
        print(f"Train accuracy: {train_accuracy}, train cer: {train_cer}")
        print(f"Test accuracy: {test_accuracy}, test cer: {test_cer}")
        print("Train decoded outputs: ", train_decoded_outputs[:4], " Train original targets: ",
              train_original_targets_list[:4])
        print("Test decoded outputs: ", test_decoded_outputs[:4], " Test original targets: ",
              test_original_targets_list[:4])
        scheduler.step(test_running_loss)


if __name__ == '__main__':
    main()

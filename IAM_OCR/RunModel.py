import numpy as np
import torch
import IAM_OCR.utils as utils


class RunModel:
    def __init__(self, epochs, device, train_loader, test_loader, decoder, classes):
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.decoder = decoder
        self.classes = classes

    def train_model(self, model, optimizer):
        model.train()
        running_loss = 0
        outputs = []
        decoded_outputs = []
        original_targets_list = []
        for images, targets, target_lengths, original_targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            batch_outputs, batch_loss = model(images, targets, target_lengths)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # cos to ma dac dla exploding gradients
            optimizer.step()
            outputs.append(batch_outputs)
            running_loss += batch_loss.item()
            original_targets_list.append(original_targets)
        decoded_outputs.append(self.decoder.decode(outputs, self.classes))

        decoded_outputs = np.concatenate(decoded_outputs)
        original_targets_list = np.concatenate(original_targets_list)
        running_loss = running_loss / len(self.train_loader)
        accuracy, cer = utils.calculate_metrics(original_targets_list, decoded_outputs)
        return decoded_outputs, running_loss, original_targets_list, accuracy, cer

    def test_model(self, model):
        model.eval()
        running_loss = 0
        outputs = []
        decoded_outputs = []
        original_targets_list = []
        with torch.no_grad():
            for images, targets, target_lengths, original_targets in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                batch_outputs, batch_loss = model(images, targets, target_lengths)
                outputs.append(batch_outputs)
                running_loss += batch_loss.item()
                original_targets_list.append(original_targets)
            decoded_outputs.append(self.decoder.decode(outputs, self.classes))

        decoded_outputs = np.concatenate(decoded_outputs)
        original_targets_list = np.concatenate(original_targets_list)
        running_loss = running_loss / len(self.test_loader)
        accuracy, cer = utils.calculate_metrics(original_targets_list, decoded_outputs)
        return decoded_outputs, running_loss, original_targets_list, accuracy, cer

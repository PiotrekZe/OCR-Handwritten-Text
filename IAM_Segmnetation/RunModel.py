import torch


class RunModel:
    def __init__(self, epochs, device, train_loader, test_loader):
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, model, optimizer, criterion):
        model.train()
        running_loss = 0
        # output_masks = []
        # real_masks = []
        for images, masks, image_shape_x, image_shape_y, mask_shape_x, mask_shape_y in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            batch_outputs = model(images)
            batch_loss = criterion(batch_outputs, masks)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # cos to ma dac dla exploding gradients
            batch_loss.backward()
            optimizer.step()
            # output_masks.append(batch_outputs)
            running_loss += batch_loss.item()
            # real_masks.append(masks)

        running_loss = running_loss / len(self.train_loader)
        # calculate accuracy
        return running_loss, masks[0], batch_outputs[0], images[0]

    def test_model(self, model, criterion):
        model.eval()
        running_loss = 0
        # output_masks = []
        # real_masks = []
        with torch.no_grad():
            for images, masks, image_shape_x, image_shape_y, mask_shape_x, mask_shape_y in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                batch_outputs = model(images)
                batch_loss = criterion(batch_outputs, masks)
                # output_masks.append(batch_outputs)
                running_loss += batch_loss.item()
                # real_masks.append(masks)

        running_loss = running_loss / len(self.test_loader)
        # calculate accuracy
        return running_loss, masks[0], batch_outputs[0], images[0]

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CRNN(nn.Module):
    def __init__(self, size, num_chars, num_channels, device, dims, num_layers):
        super(CRNN, self).__init__()
        self.size = size
        self.num_chars = num_chars + 1
        self.device = device
        self.num_channels = num_channels
        self.dims = dims
        self.num_layers = num_layers

        # encoder part
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        if self.num_channels == 1:
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        cnn_input_size = self.__calculate_cnn_input_size()
        self.linear = nn.Sequential(
            nn.Linear(cnn_input_size, self.dims),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # decoder part
        self.rnn = nn.GRU(self.dims, self.dims // 2, bidirectional=True, num_layers=self.num_layers, batch_first=True)
        self.output = nn.Linear(self.dims, self.num_chars)

    def __encode(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        return x

    def __calculate_cnn_input_size(self):
        tmp_input = torch.rand(1, self.num_channels, self.size[0], self.size[1])
        x = self.cnn.conv1(tmp_input)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        return x.shape[-1]

    def __ctc_loss(self, features, targets, target_lengths):
        input_lengths = torch.full(size=(features.size(1),), fill_value=features.size(0), dtype=torch.int32)
        # target_lengths = torch.full(size=(features.size(1),), fill_value=targets.size(1), dtype=torch.int32)
        loss = nn.CTCLoss(blank=0)(features, targets, input_lengths, target_lengths)
        return loss

    def forward(self, images, targets, target_lengths):
        cnn_output = self.__encode(images)
        rnn_output, _ = self.rnn(cnn_output)
        linear_output = self.output(rnn_output)
        x = linear_output.permute(1, 0, 2)
        x = torch.nn.functional.log_softmax(x, 2)
        loss = self.__ctc_loss(x, targets, target_lengths)
        return x, loss

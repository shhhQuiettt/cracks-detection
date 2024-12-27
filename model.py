import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channel: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=3),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3),
        )

    def forward(self, X):
        return self.net(X)


class Encoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        assert channels[0] < channels[1]

        self.blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        downscaled_features = []
        for block in self.blocks:
            X = block(X)
            downscaled_features.append(X)
            X = self.pool(X)

        return downscaled_features


class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        assert channels[0] > channels[1]

        super().__init__()

        self.blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.up_convolution = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1], kernel_size=2, stride=2
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, X, downscaled_features):
        for i, block in enumerate(self.blocks):
            X = self.up_convolution[i](X)

            # Adding the previously downscaled feature
            feature = downscaled_features[i]
            _, _, h, w = X.shape
            cropped_feature = torchvision.transforms.CenterCrop([h, w])(feature)
            X = torch.cat([X, cropped_feature], dim=1)

            X = block(X)

        return X


class UNet(nn.Module):
    def __init__(
        self, encoding_channels: tuple[int, ...], decoding_channels: tuple[int, ...], retain_dim:bool =False, output_format=(448,448)
    ):
        super().__init__()
        self.encoder = Encoder(encoding_channels)
        self.decoder = Decoder(decoding_channels)
        self.head_out = nn.Conv2d(decoding_channels[-1], 1, kernel_size=1)
        self.retain_dim  = retain_dim
        self.output_format = output_format

    def forward(self, X):
        features = self.encoder(X)
        X = self.decoder(features[-1], features[::-1][1:])
        X = self.head_out(X)
        if self.retain_dim:
            X = F.interpolate(X, self.output_format)
        return X


class Trainer:
    def __init__(
        self,
        *,
        model,
        loss_function,
        train_loader,
        val_loader,
        optimizer,
    ):
        self.model = model.to('cuda')
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')

            y_batch_prediced = self.model(X_batch)
            loss = self.loss_function(y_batch, y_batch_prediced)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)

        return epoch_loss
    
    def validate(self):
        self.model.eval()
        running_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to('cuda')
                y_batch = y_batch.to('cuda')

                y_batch_prediced = self.model(X_batch)
                loss = self.loss_function(y_batch, y_batch_prediced)

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.val_loader)

        return epoch_loss
    
    def train(self, epochs: int):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            print("fff")
            val_loss = self.validate()

            print(f"Epoch: {epoch} - Train Loss: {train_loss} - Val Loss: {val_loss}")
        



# endoding_channels = (3, 64, 128, 256, 512, 1024)
# decoding_channels = (1024, 512, 256, 128, 64)

# model = UNet(endoding_channels, decoding_channels, retain_dim=True)

# encoder = Encoder(endoding_channels)
# dec = Decoder(decoding_channels)

# X = torch.randn((2, 3, 448, 448))

# output = model(X)
# print(output.shape)

# features = encoder(X)
# print(features[-1].shape)
# res = dec(features[-1], features[::-1][1:])

# print(res.shape)

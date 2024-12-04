import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConv(nn.Module):
    def __init__(self, in_channels, out_channels, ks, p):
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        super(UNetConv, self).__init__()
        self._model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=ks, padding=ks // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(p=p),
            nn.Conv1d(out_channels, out_channels, kernel_size=ks, padding=ks // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, X):
        return self._model(X)


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, ks, p):
        super(UNetDown, self).__init__()
        self._model = nn.Sequential(
            nn.MaxPool1d(2),
            UNetConv(in_channels, out_channels, ks, p)
        )

    def forward(self, X):
        return self._model(X)


class UNetUp(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels, ks, p):
        super(UNetUp, self).__init__()
        in_channels = int(in_channels)
        in_channels_skip = int(in_channels_skip)
        out_channels = int(out_channels)

        self._up = nn.ConvTranspose1d(in_channels, in_channels,
                                      kernel_size=ks - 1,
                                      stride=2,
                                      padding=(ks - 1) // 2 - 1)
        self._model = UNetConv(in_channels + in_channels_skip, out_channels, ks, p)

    def forward(self, X_skip, X):
        X = self._up(X)
        diff = X_skip.size()[2] - X.size()[2]
        X = F.pad(X, (diff // 2, diff - diff // 2))
        return self._model(torch.cat([X_skip, X], dim=1))


class Encoder(nn.Module):
    def __init__(self, in_channels, channels_coeff=1, q=2, kernel_size=23, p=0.1):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.q = q
        self.p = p
        self._input = UNetConv(q ** 0 * self.in_channels, q ** 1 * in_channels, kernel_size, p)
        self._down1 = UNetDown(q ** 1 * self.in_channels, q ** 2 * self.in_channels, kernel_size, p)
        self._down2 = UNetDown(q ** 2 * self.in_channels, q ** 3 * self.in_channels, kernel_size, p)
        self._down3 = UNetDown(q ** 3 * self.in_channels, q ** 4 * self.in_channels, kernel_size, p)
        self._down4 = UNetDown(q ** 4 * self.in_channels, q ** 5 * self.in_channels, kernel_size, p)
        self._down5 = UNetDown(q ** 5 * self.in_channels, q ** 6 * self.in_channels, kernel_size, p)

    def forward(self, x):
        x1 = self._input(x)
        x2 = self._down1(x1)
        x3 = self._down2(x2)
        x4 = self._down3(x3)
        x5 = self._down4(x4)
        return x1, x2, x3, x4, x5, self._down5(x5)


class Decoder(nn.Module):
    def __init__(self, encoder: Encoder, num_classes, reshape=False):
        super(Decoder, self).__init__()
        self.encoder = encoder
        self._up1 = UNetUp(encoder.q ** 6 * encoder.in_channels,
                           encoder.q ** 5 * encoder.in_channels,
                           encoder.q ** 5 * encoder.in_channels,
                           encoder.kernel_size,
                           encoder.p)

        self._up2 = UNetUp(encoder.q ** 5 * encoder.in_channels,
                           encoder.q ** 4 * encoder.in_channels,
                           encoder.q ** 4 * encoder.in_channels,
                           encoder.kernel_size,
                           encoder.p)

        self._up3 = UNetUp(encoder.q ** 4 * encoder.in_channels,
                           encoder.q ** 3 * encoder.in_channels,
                           encoder.q ** 3 * encoder.in_channels,
                           encoder.kernel_size,
                           encoder.p)

        self._up4 = UNetUp(encoder.q ** 3 * encoder.in_channels,
                           encoder.q ** 2 * encoder.in_channels,
                           encoder.q ** 2 * encoder.in_channels,
                           encoder.kernel_size,
                           encoder.p)

        self._up5 = UNetUp(encoder.q ** 2 * encoder.in_channels,
                           encoder.q ** 1 * encoder.in_channels,
                           num_classes,
                           encoder.kernel_size,
                           encoder.p)

        self._output = nn.Conv1d(num_classes, num_classes, kernel_size=1)
        self.reshape = reshape
        self.num_classes = num_classes

    def forward(self, x1, x2, x3, x4, x5, x):
        batch_size = len(x)
        x = self._up1(x5, x)
        x = self._up2(x4, x)
        x = self._up3(x3, x)
        x = self._up4(x2, x)
        x = self._up5(x1, x)
        x = self._output(x)
        if self.reshape:
            x = x.reshape(batch_size, 4, 12, -1)
        return x


class UNet(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(UNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(*self.encoder(x))

    def log(self):
        return f"UNet(in_channels={self.encoder.in_channels}, num_classes={self.decoder.num_classes}, " \
               f"q={self.encoder.q}, reshape={self.decoder.reshape}, kernel_size={self.encoder.kernel_size})"
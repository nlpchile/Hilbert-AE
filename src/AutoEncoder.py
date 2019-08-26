import torch


# General
class Encoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        super(Encoder, self).__init__()

        self.model = model(**kwargs)

    def forward(self, x):
        z = self.model(x)
        return z


class Decoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        super(Decoder, self).__init__()

        self.model = model(**kwargs)

    def forward(self, z):
        y_hat = self.model(z)
        return y_hat


class Autoencoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module,
                 **kwargs) -> None:
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x=x)
        x_hat = self.decoder(z=z)
        return x_hat, z


# Particular
class convolutional_encoder(torch.nn.Module):
    def __init__(self, nc: int, ndf: int, **kwargs) -> None:
        super(convolutional_encoder, self).__init__()

        self.nc = nc
        self.ndf = ndf

        self.kernel_size: int = 4
        self.stride: int = 2
        self.padding: int = 1
        self.bias: bool = False

        # Activation
        self.inplace: bool = True

        self.model = torch.nn.Sequential(
            # input is (nc) x 32 x 32
            torch.nn.Conv2d(in_channels=self.nc,
                            out_channels=self.ndf,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            bias=self.bias),
            torch.nn.ReLU(inplace=self.inplace),

            # state size. (ndf) x 16 x 16
            torch.nn.Conv2d(in_channels=self.ndf,
                            out_channels=self.ndf * 2,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            bias=self.bias),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.ReLU(inplace=self.inplace),

            # state size. (ndf*2) x 8 x 8
            torch.nn.Conv2d(in_channels=self.ndf * 2,
                            out_channels=self.ndf * 4,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            bias=self.bias),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.ReLU(inplace=self.inplace),

            # state size. (ndf*4) x 4 x 4
            torch.nn.Conv2d(in_channels=self.ndf * 4,
                            out_channels=self.ndf * 8,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            bias=self.bias),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class convolutional_decoder(torch.nn.Module):
    def __init__(self, nc: int, ndf: int, **kwargs) -> None:
        super(convolutional_decoder, self).__init__()

        self.nc = nc
        self.ndf = ndf

        self.kernel_size: int = 4
        self.stride: int = 2
        self.padding: int = 1
        self.bias: bool = False

        # Activation
        self.inplace: bool = True

        self.model = torch.nn.Sequential(
            # state size. (ndf*4) x 4 x 4
            torch.nn.ConvTranspose2d(in_channels=self.ndf * 8,
                                     out_channels=self.ndf * 4,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=self.bias),
            torch.nn.BatchNorm2d(num_features=self.ndf * 4),
            torch.nn.ReLU(inplace=self.inplace),

            # state size. (ndf*2) x 8 x 8
            torch.nn.ConvTranspose2d(in_channels=self.ndf * 4,
                                     out_channels=self.ndf * 2,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=self.bias),
            torch.nn.BatchNorm2d(num_features=self.ndf * 2),
            torch.nn.ReLU(inplace=self.inplace),

            # state size. (ndf) x 16 x 16
            torch.nn.ConvTranspose2d(in_channels=self.ndf * 2,
                                     out_channels=self.ndf,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=self.bias),
            torch.nn.BatchNorm2d(num_features=self.ndf),
            torch.nn.ReLU(inplace=self.inplace),

            # input is (nc) x 32 x 32
            torch.nn.ConvTranspose2d(in_channels=self.ndf,
                                     out_channels=self.nc,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=self.bias),
        )

    def forward(self, x):
        out = self.model(x)
        return out


# Still compatible with older implementation
class autoencoder(torch.nn.Module):
    def __init__(self, nc, ndf, **kwargs):
        super(autoencoder, self).__init__()

        self.nc = nc
        self.ndf = ndf

        self.encoder = convolutional_encoder(nc=self.nc,
                                             ndf=self.ndf,
                                             **kwargs)

        self.decoder = convolutional_decoder(nc=self.nc,
                                             ndf=self.ndf,
                                             **kwargs)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

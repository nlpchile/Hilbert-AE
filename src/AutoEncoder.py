"""Autoencoder Modules."""
from typing import Any, Dict, Tuple

import torch

Tensor = torch.Tensor

try:
    from apex import amp
    APEX_IS_AVAILABLE = True
except ImportError:
    APEX_IS_AVAILABLE = False
    pass


# Training
def training_step(model: torch.nn.Module, x: Tensor,
                  optimizer: torch.optim.Optimizer,
                  criterion: torch.nn.Module) -> Dict[str, Any]:
    """
    Perform one optimization step over an input batch.

    Args:
        model (torch.nn.Module) : A torch model.

        x (Tensor) :  An input torch tensor.

        optimizer (torch.optim) : A torch optimizer.

        criterion (torch.nn) : A torch criterion / loss.

    Returns:
        (Dict[str, Any]) : A dict containing the results.

                                keys : "loss", "x_hat" and "z"
                                values: float, Tensor, Tensor

    """
    # x : [batch_size, *input_size]

    # Training Fase
    model.train()

    # Zero Grad
    optimizer.zero_grad()

    # Forward

    # [batch_size, *input_size], [batch_size, *latent_size]
    x_hat, z = model.forward(x)

    # Loss
    loss = criterion(x_hat, x)

    # Backward
    if APEX_IS_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            # Gradients are unscaled during context manager exit.

        #  Now it's safe to clip.  Replace
        #       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)

        # with
        #       torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)

        # or
        #       torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), max_)

    else:
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)

    # Step
    optimizer.step()

    output = {"loss": loss.item(), "x_hat": x_hat, "z": z}

    return output


# Validation
def validation_step(model: torch.nn.Module, x: Tensor,
                    criterion: torch.nn.Module) -> Dict[str, Any]:
    """
    Perform an evaluation step over an input batch.

    Args:
        model (torch.nn.Module) : A torch model.

        x (Tensor) :  An input torch tensor.

        criterion (torch.nn) : A torch criterion / loss.

    Returns:
        (Dict[str, Any]) : A dict containing the results.

                                keys : "loss", "x_hat" and "z"
                                values: float, Tensor, Tensor

    """
    # x : [batch_size, *input_size]

    # evaluation fase
    model.eval()

    # To avoid gradients computation
    with torch.no_grad():

        # Forward

        # [batch_size, *input_size], [batch_size, *latent_size]
        x_hat, z = model.forward(x)

        # Loss
        loss = criterion(x_hat, x)

        output = {"loss": loss.item(), "x_hat": x_hat, "z": z}

        return output


# General
class Encoder(torch.nn.Module):
    """A general Encoder Class."""

    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        """
        Encoder.

        Args:
            model [torch.nn.Module] : A torch model.

        """
        super(Encoder, self).__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """Encode an input x to a latent representation z."""
        z = self.model(x)
        return z

    # TODO : Check this idea
    # @property
    # def shapes(self):
    #     """Input and  Output Shape"""
    #     return (self.model.input_shape, self.model_output_shape)


class Decoder(torch.nn.Module):
    """A general Decoder Class."""

    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        """
        Decoder.

        Args:
            model [torch.nn.Module] : A torch model.

        """
        super(Decoder, self).__init__()

        self.model = model

    def forward(self, z: Tensor) -> Tensor:
        """Decode an input representation z into a representation y_hat."""
        y_hat = self.model(z)
        return y_hat


class Autoencoder(torch.nn.Module):
    """A general Autoencoder Class."""

    def __init__(self, encoder: Encoder, decoder: Decoder, **kwargs) -> None:
        """
        Autoencoder.

        Args:
            encoder [Encoder] : An Encoder model.

            decoder [Decoder] : A Decoder model.

        """
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Represent an input in an encoded space, then reconstructs it from it.

        Args:
            x [Tensor] : An input torch tensor.

        Returns:
            (Tuple[Tensor]) : A Tuple containing reconstructed input and
                              its internal latent representation.

        """
        # Latent representation
        z = self.encoder(x=x)

        # Reconstructed input
        x_hat = self.decoder(z=z)

        return x_hat, z

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode an input x to its latent representation z.

        Args:
            x (Tensor) : An input torch tensor.

        Returns:
            (Tensor) : An encoded latent representation.

        """
        z = self.encoder(x=x)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode a latent representation z into a reconstructed space x_hat.

        Args:
            z (Tensor) : An latent representation.

        Returns:
            (Tensor) : A reconstructed space x_hat.

        """
        x_hat = self.decoder(z=z)
        return x_hat


# Particular
class convolutional_encoder(Encoder):
    """Convolutional Encoder."""

    def __init__(self, nc: int, ndf: int, **kwargs) -> None:
        """
        Convolutional Encoder.

        Args:
            nc (int) : Input Channels

            ndf (int) : Output Channels.

        """

        self.nc = nc
        self.ndf = ndf

        self.kernel_size: int = 4
        self.stride: int = 2
        self.padding: int = 1
        self.bias: bool = False

        # Activation
        self.inplace: bool = True

        model = torch.nn.Sequential(

            # input is (nc) x 8 x 8
            torch.nn.Upsample(size=(32, 32),
                              scale_factor=None,
                              mode="bilinear",
                              align_corners=False),
            torch.nn.ReLU(inplace=self.inplace),

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
            torch.nn.BatchNorm2d(num_features=ndf * 2),
            torch.nn.ReLU(inplace=self.inplace),

            # state size. (ndf*2) x 8 x 8
            torch.nn.Conv2d(in_channels=self.ndf * 2,
                            out_channels=self.ndf * 4,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            bias=self.bias),
            torch.nn.BatchNorm2d(num_features=ndf * 4),
            torch.nn.ReLU(inplace=self.inplace),

            # state size. (ndf*4) x 4 x 4
            torch.nn.Conv2d(in_channels=self.ndf * 4,
                            out_channels=self.ndf * 8,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            bias=self.bias),
        )

        super(convolutional_encoder, self).__init__(model=model)


class convolutional_decoder(Decoder):
    """Convolutional Decoder."""

    def __init__(self, nc: int, ndf: int, **kwargs) -> None:
        """

        Convolutional Decoder.

        Args:
            nc (int) : Input Channels

            ndf (int) : Output Channels.

        """
        # super(convolutional_decoder, self).__init__()

        self.nc = nc
        self.ndf = ndf

        self.kernel_size: int = 4
        self.stride: int = 2
        self.padding: int = 1
        self.bias: bool = False

        # Activation
        self.inplace: bool = True

        model = torch.nn.Sequential(
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
            # input is (nc) x 32 x 32
            torch.nn.Upsample(size=(8, 8),
                              scale_factor=None,
                              mode="bilinear",
                              align_corners=False))

        super(convolutional_decoder, self).__init__(model=model)


# Still compatible with older implementation
class autoencoder(Autoencoder):
    """A convolutional autoencoder."""

    def __init__(self, nc: int, ndf: int, **kwargs):
        """
        Convolutional Autoencoder.

        Args:
            nc (int) : Input Channels

            ndf (int) : Output Channels.

        """

        self.nc = nc
        self.ndf = ndf

        encoder = convolutional_encoder(nc=self.nc, ndf=self.ndf, **kwargs)

        decoder = convolutional_decoder(nc=self.nc, ndf=self.ndf, **kwargs)

        super(autoencoder, self).__init__(encoder=encoder, decoder=decoder)


class Reshape(torch.nn.Module):
    def __init__(self, shape: Tuple[int, ...], **kwargs) -> None:
        super(Reshape, self).__init__()

        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.shape)


class simple_encoder(Encoder):
    def __init__(self, num_channels: int, **kwargs) -> None:

        self.num_channels = num_channels

        reduce = torch.nn.Conv2d(in_channels=self.num_channels,
                                 out_channels=128,
                                 kernel_size=(1, 1),
                                 stride=1,
                                 padding=0,
                                 bias=True)

        conv1 = torch.nn.Conv2d(in_channels=128,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=True)

        batchnorm1 = torch.nn.BatchNorm2d(num_features=64)

        conv2 = torch.nn.Conv2d(in_channels=64,
                                out_channels=32,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=True)

        batchnorm2 = torch.nn.BatchNorm2d(num_features=32)

        conv3 = torch.nn.Conv2d(in_channels=32,
                                out_channels=16,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=True)

        batchnorm3 = torch.nn.BatchNorm2d(num_features=16)

        reshape = Reshape(shape=(-1, 16))

        linear = torch.nn.Linear(in_features=16, out_features=16, bias=True)

        relu = torch.nn.ReLU(inplace=True)

        model = torch.nn.Sequential(reduce, relu, conv1, batchnorm1, relu,
                                    conv2, batchnorm2, relu, conv3, batchnorm3,
                                    relu, reshape, linear)

        super(simple_encoder, self).__init__(model=model)


class simple_decoder(Decoder):
    def __init__(self, num_channels: int, **kwargs) -> None:

        self.num_channels = num_channels

        reshape = Reshape(shape=(-1, 16, 1, 1))

        upsample = torch.nn.Upsample(size=(4, 4),
                                     scale_factor=None,
                                     mode="bilinear",
                                     align_corners=False)

        conv1 = torch.nn.ConvTranspose2d(in_channels=16,
                                         out_channels=32,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=True)

        batchnorm1 = torch.nn.BatchNorm2d(num_features=32)

        conv2 = torch.nn.ConvTranspose2d(in_channels=32,
                                         out_channels=64,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=True)

        batchnorm2 = torch.nn.BatchNorm2d(num_features=64)

        conv3 = torch.nn.ConvTranspose2d(in_channels=64,
                                         out_channels=128,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=True)

        batchnorm3 = torch.nn.BatchNorm2d(num_features=128)

        downsample = torch.nn.Upsample(size=(8, 8),
                                       scale_factor=None,
                                       mode="bilinear",
                                       align_corners=False)

        increase = torch.nn.Conv2d(in_channels=128,
                                   out_channels=self.num_channels,
                                   kernel_size=(1, 1),
                                   stride=1,
                                   padding=0,
                                   bias=True)

        relu = torch.nn.ReLU(inplace=True)

        model = torch.nn.Sequential(reshape, upsample, relu, conv1, batchnorm1,
                                    relu, conv2, batchnorm2, relu, conv3,
                                    batchnorm3, relu, downsample, relu,
                                    increase)

        super(simple_decoder, self).__init__(model=model)


class simple_autoencoder(Autoencoder):
    def __init__(self, num_channels: int, **kwargs) -> None:

        self.num_channels = num_channels

        encoder = simple_encoder(num_channels=self.num_channels)
        decoder = simple_decoder(num_channels=self.num_channels)

        super(simple_autoencoder, self).__init__(encoder=encoder,
                                                 decoder=decoder)

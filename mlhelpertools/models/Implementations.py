import functools
import itertools
import random
from typing import Optional, List, Literal

import torch
import torchvision.models


class ImagePool:
    """
    With probability `p`, return an image from cache instead of passed image.
    """

    def __init__(self, capacity: int = 100, p: float = 0.5):
        assert 0 < p < 1.0, "Invalid probability value, expected ]0,1], found: {}".format(p)
        assert capacity >= 0, "Invalid capacity {}".format(capacity)
        self.capacity = capacity
        self.p = p
        self.buffer = list()

    def __call__(self, x):
        if self.capacity == 0:
            return x

        output = list()

        for image in x:
            image = torch.unsqueeze(image, 0)
            if len(self.buffer) < self.capacity:
                self.buffer.append(image)
                output.append(image)
            else:
                if random.uniform(0, 1) > self.p:
                    # swap
                    i = random.randint(0, self.capacity - 1)
                    tmp = self.buffer[i].clone()
                    self.buffer[i] = image
                    output.append(tmp)
                else:
                    output.append(image)

        return torch.cat(output, 0)


class CycleGAN(torch.nn.Module):
    def __init__(self, generatorA: Optional[torch.nn.Module], generatorB: Optional[torch.nn.Module],
                 classifierA: Optional[torch.nn.Module], classifierB: Optional[torch.nn.Module], lambda_A: float = None,
                 idtA: float = 0.1, idtB: float = 0.1, loss_C: torch.nn.Module = torch.nn.L1Loss,
                 loss_G: torch.nn.Module = torch.nn.MSELoss, optimizer_G: torch.optim = torch.optim.Adam,
                 optimizer_C: torch.optim = torch.optim.Adam, poolA: Optional[ImagePool] = None,
                 poolB: Optional[ImagePool] = None, channels_in: Optional[int] = None,
                 channels_out: Optional[int] = None):
        super(CycleGAN, self).__init__()

        if generatorA is not None:
            assert channels_in is not None, "When given a generator, number of channels is expected."
        if generatorB is not None:
            assert channels_out is not None, "When given a generator, number of channels is expected."

        self.netAB = ResnetGenerator(channels_in, channels_out) if generatorA is None else generatorA
        self.netBA = ResnetGenerator(channels_out, channels_in) if generatorB is None else generatorB
        self.discA = ConvMLP(channels_in, [10, 5], (2, 2), 1, 50, True) if classifierA is None else classifierA
        self.discB = ConvMLP(channels_out, [10, 5], (2, 2), 1, 50, True) if classifierB is None else classifierB

        self.idtAA = idtA
        self.idtBB = idtB

        self.loss_classifier = loss_C()
        self.loss_generator = loss_G()

        self.optimizer_generator = optimizer_G(itertools.chain(self.netAB.parameters(), self.netBA.parameters()))
        self.optimizer_classifier = optimizer_C(itertools.chain(self.discA.parameters(), self.discB.parameters()))

        self.lambda_A = lambda_A
        self.lambda_B = 1 - lambda_A

        self.poolA = ImagePool(50) if poolA is None else poolA
        self.poolB = ImagePool(50) if poolB is None else poolB

    def backward_gen(self, original, reconstructed, pred_gen, idt, factor: float, factor_idt: float):
        loss_a = self.loss_classifier(pred_gen, torch.ones_like(pred_gen))
        loss_rec = self.loss_generator(original, reconstructed) * factor
        loss_idt = 0

        if factor_idt != 0:
            loss_idt = self.loss_generator(original, idt) * factor * factor_idt

        return loss_a + loss_rec + loss_idt

    def backward_disc(self, t, f):
        return (self.loss_classifier(t, torch.ones_like(t)) + self.loss_classifier(f, torch.zeros_like(f))) * 0.5

    def train_step(self, a, b):
        fakeA = self.netBA(b)
        fakeB = self.netAB(a)
        recA = self.netBA(fakeB)
        recB = self.netAB(fakeA)
        idtA = self.netBA(a)
        idtB = self.netAB(b)

        predA_true = self.discA(a)
        predA_gen = self.discA(fakeA)

        predB_true = self.discB(b)
        predB_gen = self.discB(fakeB)

        self.__set_require_grad(self.discA, False)
        self.__set_require_grad(self.discB, False)

        self.optimizer_generator.zero_grad()

        genA_loss = self.backward_gen(a, recA, predA_gen, idtA, self.lambda_A, self.idtAA)
        genB_loss = self.backward_gen(b, recB, predB_gen, idtB, self.lambda_B, self.idtBB)

        (genA_loss + genB_loss).backward()

        self.optimizer_generator.step()

        self.__set_require_grad(self.discA, True)
        self.__set_require_grad(self.discB, True)

        self.optimizer_classifier.zero_grad()

        predA_gen = self.poolA(predA_gen)
        predB_gen = self.poolB(predB_gen)

        discA_loss = self.backward_disc(predA_true, predA_gen).backward()
        discB_loss = self.backward_disc(predB_true, predB_gen).backward()

        self.optimizer_classifier.step()

        return genA_loss, genB_loss, discA_loss, discB_loss

    def forward(self, x: torch.Tensor | tuple):
        a, b = None, None

        if isinstance(x, tuple):
            if len(tuple) == 2:
                a, b = x
            else:
                raise ValueError("Invalid tuple format, expected (tensor, tensor) found length: {}".format(len(x)))

        if x.ndim == 4 and x.shape[0] == 2:
            a, b = x
        elif x.ndim == 4 and x.shape[0] == 1:
            if not self.training:
                a = x
            else:
                raise ValueError("Shape of tensor must be (2, batch, channel, width, height) for training, "
                                 "found : {}".format(x.shape))
        else:
            raise ValueError("Invalid number of dimansions expercted (2, batch, channel, width, height) or "
                             "(1, batch, channel, width, height) found: {}".format(x.shape))

        fakeA = None
        recB = None
        idtB = None
        discB = None

        fakeB = self.netAB(a)
        recA = self.netBA(fakeB)
        idtA = self.netBA(a)
        discA = self.discA(a)

        if b is not None:
            fakeA = self.netBA(b)
            recB = self.netAB(fakeA)
            idtB = self.netAB(b)
            discB = self.discB(b)

        return fakeA, fakeB, recA, recB, idtA, idtB, discA, discB

    def __set_require_grad(self, model: torch.nn.Module, requires_grad: bool):
        for p in model.parameters():
            p.requires_grad = requires_grad


class ResnetGenerator(torch.nn.Module):
    """
    Resnet-based generator copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=torch.nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d

        model = [torch.nn.ReflectionPad2d(3),
                 torch.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 torch.nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      torch.nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                               kernel_size=3, stride=2,
                                               padding=1, output_padding=1,
                                               bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      torch.nn.ReLU(True)]
        model += [torch.nn.ReflectionPad2d(3)]
        model += [torch.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [torch.nn.Tanh()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(torch.nn.Module):
    """Define a Resnet block also copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim), torch.nn.ReLU(True)]
        if use_dropout:
            conv_block += [torch.nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [torch.nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [torch.nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [torch.nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return torch.nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class MLP(torch.nn.Module):
    """
    Simple Multi layered perceptron for easy classification tasks
    """

    def __init__(self, input_n: int, hidden_layers: List[int], output_n: int, use_bias: bool = True,
                 activation: torch.nn.Module = torch.nn.ReLU, activation_out: torch.nn.Module = torch.nn.Sigmoid):
        super(MLP, self).__init__()

        self.model = list()

        self.model.append(torch.nn.Linear(input_n, hidden_layers[0], bias=use_bias))
        self.model.append(activation())

        for i in range(len(hidden_layers) - 1):
            self.model.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1], bias=use_bias))
            self.model.append(activation())

        self.model.append(torch.nn.Linear(hidden_layers[-1], output_n, bias=use_bias))
        self.model.append(activation_out())

        self.model = torch.nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ConvMLP(torch.nn.Module):
    """
    Convolutional Multi-Layered Perceptron for classification tasks.
    """

    def __init__(self, input_c: int, channels: List[int], kernels: List[int] | tuple[int, int], output_channels: int,
                 n_out: int, use_bias: bool, batch_norm: Optional[torch.nn.Module] = torch.nn.BatchNorm2d,
                 activation: torch.nn.Module = torch.nn.ReLU):
        super(ConvMLP, self).__init__()

        self.conv = list()

        self.conv.append(torch.nn.Conv2d(input_c, channels[0], bias=use_bias, kernel_size=kernels[0]))
        self.conv.append(activation())

        for i in range(len(channels) - 1):
            self.conv.append(torch.nn.Conv2d(channels[i], channels[i + 1], bias=use_bias,
                                             kernel_size=kernels[i] if isinstance(kernels, List) else kernels)
                             )
            self.conv.append(activation())

            if batch_norm is not None:
                self.conv.append(batch_norm())

        self.conv = torch.nn.Sequential(*self.conv)
        self.linear = MLP(output_channels, [512, 256], n_out)

    def forward(self, x):
        x = self.conv(x).view(x.shape[0], -1)
        return self.linear(x)


class SiameseNetwork(torch.nn.Module):
    """
    https://github.com/pytorch/examples/blob/main/siamese_network/main.py
    """

    def __init__(self, backbone: Optional[torch.nn.Module], classifier: Optional[torch.nn.Module],
                 feature_union_method: Literal["cat", "bilinear", "bilinear_multi", "diff"] = "cat",
                 backbone_output_shape: Optional[int] = None):
        super(SiameseNetwork, self).__init__()
        METHODS = ["cat", "diff", "bilinear", None]

        assert feature_union_method in METHODS, (
                "Invalid union method, expected {} " +
                "but found {}".format(METHODS, feature_union_method)
        )

        if backbone is not None and classifier is None and feature_union_method is not None:
            assert backbone_output_shape is not None, "When backbone is not None, `backbone_output_shape` is expected."

        self.backbone = None
        self.classifier = None
        temp_fc_in = None

        if backbone is None:
            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            temp_fc_in = resnet.fc.in_features
            self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        else:
            self.backbone = backbone
            temp_fc_in = backbone_output_shape

        if feature_union_method is not None:
            if classifier is None:
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(temp_fc_in * 2, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 1),
                    torch.nn.Sigmoid()
                )
            else:
                self.classifier = classifier

        self.union_method = feature_union_method

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        features = self.forward_backbone(x, y)

        output = self.classifier(features) if self.classifier is not None else features

        return output

    def forward_backbone(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        features_x = self.backbone(x)
        features_y = self.backbone(y)

        features = None

        if self.union_method == "cat":
            features = torch.cat((features_x, features_y), dim=1)
        elif self.union_method == "bilinear":  # TODO validate
            # (batch, channel, width, height) x (batch, channel, width, height) = (b, channel, channel)
            features = torch.einsum("bijk,bljk->bil", features_x, features_y)  # batch,channel matrix mult
            features /= (features_x.shape[2] * features_x.shape[3])

            features = torch.multiply(torch.sign(features), torch.sqrt(torch.abs(features)))

            features = features / torch.sqrt(torch.max(torch.sum(features ** 2), torch.tensor(1e-8)))  # l2 norm
        elif self.union_method == "diff":
            features = torch.abs(features_x - features_y)
        elif self.union_method is None:
            features = features_x, features_y

        return features

    class ResidualBlock(torch.nn.Module):
        def __init__(self, block: torch.nn.Module, merge_method: Literal["add", "concat"]):
            super().__init__()
            self.block = block
            self.merge_method = merge_method

        def forward(self, x):
            output = None

            if self.merge_method == "add":
                output = x + self.block(x)
            elif self.merge_method == "concat":
                output = torch.cat(x, self.block(x))

            return output


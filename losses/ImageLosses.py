import geomloss
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Optional, Dict, Literal, Tuple, Any


class WassersteinLoss(torch.nn.Module):
    """
    Earth mover distance loss from https://www.kernel-operations.io/geomloss
    """

    def __init__(self, format_channels: str = "WHC", reduction: bool = True, **kwargs):
        super().__init__()
        assert "C" in format_channels and "W" in format_channels and "H" in format_channels and len(format_channels) == 3, (
            "Missing element in format, expected `C` and `W` and `H`, found {}".format(format_channels)
        )
        sl_args = {
            "blur": 0.001
        }

        sl_args |= kwargs

        self.loss_module = geomloss.SamplesLoss(**sl_args)
        self.format = format_channels
        self.reduction = reduction

    def to_pointcloud(self, x):
        b, h, w, c = 1, 1, 1, 1

        if len(x.size()) == 4:
            if self.format == "WHC" or self.format == "HWC":
                b, h, w, c = x.size()
            elif self.format == "CWH" or self.format == "CHW":
                b, c, h, w = x.size()
            else:
                b, h, c, w, = x.size()
        elif len(x.size()) == 3:
            if self.format == "WHC" or self.format == "HWC":
                h, w, c = x.size()
            elif self.format == "CWH" or self.format == "CHW":
                c, h, w = x.size()
            else:
                h, c, w = x.size()

        return x.view(b, w * h, c)

    def forward(self, generated, target):
        pc_g = self.to_pointcloud(generated)
        pc_t = self.to_pointcloud(target)

        if self.reduction:
            return self.loss_module(pc_g, pc_t).sum()
        else:
            return self.loss_module(pc_g, pc_t)


class PerceptualLoss(torch.nn.Module):
    """
    Implements PerceptualLoss as explained in https://arxiv.org/pdf/1603.08155
    Setting `x_factor = 0` allows to unload that compleately.
    """

    def __init__(self, discriminator_network: torch.nn.Module = None, features_dict: Optional[Dict[str, str]] = None,
                 content_dict: Optional[Dict[str, str]] = None, reduction: bool = True, tv_factor: float = 1.0,
                 style_factor: float = 1.0, pixel_factor: float = 0.0, content_factor: float = 1.0,
                 needs_norm: bool = True, custom_transforms: Optional[torchvision.transforms.Compose] = None):
        super().__init__()

        if discriminator_network is None:
            self.model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.DEFAULT).features.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            self.model = discriminator_network

        if needs_norm:
            if discriminator_network is None:
                self.transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(
                        mean=[0.48235, 0.45882, 0.40784],
                        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
                    )
                ])
            else:
                self.transforms = custom_transforms

        if features_dict is None:  # default feature maps match original SR paper https://arxiv.org/pdf/1603.08155
            self.return_features = {
                "3": "first",
                "8": "second",
                "15": "third",
                "22": "fourth"
            }
        else:
            self.return_features = features_dict

        if content_dict is None:
            self.return_content = {
                "8": "first"
            }
        else:
            self.return_content = content_dict

        if style_factor != 0:
            self.feature_model = create_feature_extractor(self.model, self.return_features)

        if content_factor != 0:
            self.content_model = create_feature_extractor(self.model, self.return_content)

        self.reduction = reduction
        self.content_factor = content_factor if content_factor is not None else 0
        self.style_factor = style_factor if style_factor is not None else 0
        self.needs_norm = needs_norm
        self.tv_factor = tv_factor if tv_factor is not None else 0
        self.pixel_factor = pixel_factor if pixel_factor is not None else 0

        self.mse = torch.nn.MSELoss()

        if tv_factor != 0:
            self.tv = TotalVariationLoss()

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gramm matrix of tensor with size CxWxH (output from discriminator network).
        :param x: Individual tensor for feature map at layer i.
        :return: CxC Gram matrix tensor.
        """
        c, w, h = x.size()
        return torch.mm(x.view(c, w * h), x.view(c, w * h).t()) / (c * h * w)

    def calc_style_loss(self, target: Dict[str, torch.Tensor], generated: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = torch.zeros(1)

        for layer_name in target.keys():
            for i, j in zip(target[layer_name], generated[layer_name]):
                output += (self.gram_matrix(i) - self.gram_matrix(j)).pow(2).sum()

        return output

    def calc_content_loss(self, original: Dict[str, torch.Tensor], generated: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = torch.zeros(1)

        for layer_name in original.keys():
            for i, j in zip(original[layer_name], generated[layer_name]):
                output += self.mse(i, j)

        return output

    def forward(self, original, target, generated):
        style_loss = 0
        content_loss = 0
        total_variation_loss = 0
        pixel_loss = 0

        if self.tv_factor != 0:
            total_variation_loss = self.tv(generated)

        if self.style_factor != 0:
            generated_features = self.feature_model(self.transforms(generated) if self.needs_norm else generated)
            target_features = self.feature_model(self.transforms(target) if self.needs_norm else target)
            style_loss = self.calc_style_loss(target_features, generated_features)

        if self.content_factor != 0:
            original_content = self.content_model(self.transforms(original) if self.needs_norm else original)
            generated_content = self.content_model(self.transforms(generated) if self.needs_norm else generated)
            content_loss = self.calc_content_loss(original_content, generated_content)

        if self.pixel_factor != 0:
            pixel_loss = self.mse(generated, target)

        if self.reduction:
            return (
                    self.style_factor * style_loss
                    + self.content_factor * content_loss
                    + self.tv_factor * total_variation_loss
                    + self.pixel_factor * pixel_loss
            )
        else:
            return style_loss, content_loss, total_variation_loss, pixel_loss


class TotalVariationLoss(torch.nn.Module):
    """
    Typically used for denoising, this loss reduces big variance zones (noise) in (typically) images.

    https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    def __init__(self, format_channels: str = "CWH", alpha: float = 1.0):
        super().__init__()
        assert "C" in format_channels and "W" in format_channels and "H" in format_channels and len(format_channels) == 3, (
            "Missing element in format, expected `C` and `W` and `H`, found {}".format(format_channels)
        )

        self.weight = alpha
        self.format = format_channels

    def forward(self, x):
        bs_img, c_img, h_img, w_img = x.size()

        a, b = 0, 0

        if self.format == "CWH" or self.format == "CHW":
            a = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
            b = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        elif self.format == "WHC" or self.format == "HWC":
            a = torch.pow(x[:, 1:, :, :] - x[:, :-1, :, :], 2).sum()
            b = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        else:  # ?
            a = torch.pow(x[:, 1:, :, :] - x[:, :-1, :, :], 2).sum()
            b = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()

        return self.weight * (a + b) / (bs_img * c_img * h_img * w_img)


class CycleGANLoss(torch.nn.Module):
    r"""
    Generic loss module for CycleGANs.

    For use with this repo\'s CycleGAN do:
    .. code-block::

        gan = models.CycleGAN()
        loss_criterion = losses.CycleGANLoss(
                    gan.netAB, gan.netBA, gan.discA, gan.discB
                )

        # train loop
        res = gan(data)
        loss = loss_criterion(data, *res)
        loss.backward()
    """

    def __init__(self, classifier_a: Optional[torch.nn.Module], classifier_b: Optional[torch.nn.Module],
                 loss_labels: torch.nn.Module = torch.nn.L1Loss, loss_images: torch.nn.Module = torch.nn.MSELoss,
                 factor_cycle_a: float = 0.1, factor_cycle_b: float = 0.1, factor_idt_a: float = 0.001,
                 factor_idt_b: float = 0.001, reduction: Literal["sum", None] = "sum"):
        super(CycleGANLoss, self).__init__()

        self.discA = classifier_a
        self.discB = classifier_b

        self.loss_labels = loss_labels
        self.loss_images = loss_images

        self.factor_cycle_a = factor_cycle_a
        self.factor_cycle_b = factor_cycle_b
        self.factor_idt_a = factor_idt_a
        self.factor_idt_b = factor_idt_b

        self.reduction = reduction

    def backward_disc(self, t: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return (self.loss_labels(t, torch.ones_like(t)) + self.loss_labels(f, torch.zeros_like(f))) * 0.5

    def backward_gen(self, original: torch.Tensor, reconstructed: torch.Tensor, pred_gen: torch.Tensor,
                     idt: torch.Tensor, do_idt: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_g = self.loss_labels(pred_gen, torch.ones_like(pred_gen))
        loss_rec = self.loss_images(original, reconstructed)
        loss_idt = 0

        if do_idt:
            loss_idt = self.loss_images(original, idt)

        return loss_g, loss_rec, loss_idt

    def forward(self, data: torch.Tensor | tuple, fakeA: torch.Tensor, fakeB: torch.Tensor,
                reconstructedA: torch.Tensor, reconstructedB: torch.Tensor, idtA: torch.Tensor, idtB: torch.Tensor,
                discA: torch.Tensor, discB: torch.Tensor, disc_fakeA: Optional[torch.Tensor] = None,
                disc_fakeB: Optional[torch.Tensor] = None) -> torch.Tensor | \
                                                              Tuple[
                                                                  torch.Tensor, torch.Tensor, torch.Tensor,
                                                                  torch.Tensor, torch.Tensor, torch.Tensor,
                                                                  torch.Tensor, torch.Tensor
                                                              ]:
        batch_a, batch_b = None, None

        if (isinstance(data, tuple) and len(data) == 2) or (data.ndim == 4 and data.shape[0] == 2):
            batch_a, batch_b = data
        else:
            raise ValueError("Expected shape (2, batch, channels, width , height) for loss calculation, "
                             "found {}".format(data.shape))

        if (disc_fakeB is None and self.discB is None) or (disc_fakeA is None and self.discA is None):
            raise ValueError("If not given discriminators for A and B you must provide "
                             "`disc_fakeA` or `disc_fakeB` respectively.")

        self._set_disc_requires_grad(False)

        pred_genA = self.discA(fakeA) if disc_fakeA is None else disc_fakeA
        pred_genB = self.discB(fakeB) if disc_fakeB is None else disc_fakeB

        loss_g_a, loss_rec_a, loss_idt_a = self.backward_gen(batch_a, reconstructedA, pred_genA, idtA,
                                                             self.factor_idt_a != 0)

        loss_g_b, loss_rec_b, loss_idt_b = self.backward_gen(batch_b, reconstructedB, pred_genB, idtB,
                                                             self.factor_idt_b != 0)

        self._set_disc_requires_grad(True)

        c_loss_a = self.backward_disc(discA, pred_genA)
        c_loss_b = self.backward_disc(discB, pred_genB)

        output = (
            loss_g_a,
            loss_rec_a * self.factor_cycle_a,
            loss_idt_a * self.factor_idt_a,
            c_loss_a,
            loss_g_b,
            loss_rec_b * self.factor_cycle_b,
            loss_idt_b * self.factor_idt_b,
            c_loss_b)

        if self.reduction is None:
            return output
        elif self.reduction == "sum":
            return sum(output)

    def _set_disc_requires_grad(self, req_grad):
        if self.discA is not None:  # debug mode
            for p in self.discA.parameters():
                p.requires_grad = req_grad

        if self.discB is not None:
            for p in self.discB.parameters():
                p.requires_grad = req_grad

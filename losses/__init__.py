import geomloss
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Optional, Dict


class WassersteinLoss(torch.nn.Module):
    def __init__(self, format_channels: str = "WHC", **kwargs):
        super().__init__()
        assert "C" in format and "W" in format and "H" in format_channels and len(format_channels) == 3, (
            "Missing element in format, expected `C` and `W` and `H`, found {}".format(format_channels)
        )
        self.module = geomloss.SamplesLoss(**kwargs)
        self.format = format_channels

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
                h, c, w, = x.size()

        return x.view(b, c, w*h)

    def forward(self, generated, target):
        pc_g = self.to_pointcloud(generated)
        pc_t = self.to_pointcloud(target)

        return self.module(pc_g, pc_t)


class PerceptualLoss(torch.nn.Module):
    """
    Implements PerceptualLoss as explained in https://arxiv.org/pdf/1603.08155
    Setting `x_factor = 0` allows to unload that compleately.
    """
    def __init__(self, discriminator_network: torch.nn.Module = None, features_dict: Optional[Dict[str, str]] = None,
                 content_dict: Optional[Dict[str, str]] = None, return_one: bool = True, tv_factor: float = 1.0,
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
                    torchvision.transforms.Normalize(mean=[0.48235, 0.45882, 0.40784],
                                                     std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
                ])
            else:
                self.transforms = custom_transforms

        if features_dict is None:
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
                "15": "first"
            }
        else:
            self.return_content = content_dict

        if style_factor != 0:
            self.feature_model = create_feature_extractor(self.model, self.return_features)

        if content_factor != 0:
            self.content_model = create_feature_extractor(self.model, self.return_content)

        self.return_single = return_one
        self.content_factor = content_factor
        self.style_factor = style_factor
        self.needs_norm = needs_norm
        self.tv_factor = tv_factor
        self.pixel_factor = pixel_factor

        self.mse = torch.nn.MSELoss()

        if tv_factor != 0:
            self.tv = TotalVariationLoss()

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gramm matric of tensor with size CxWxH (output from discriminator network).
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
            pixel_loss = self.mse(original, target)

        if self.return_single:
            return (self.style_factor * style_loss
                    + self.content_factor * content_loss
                    + self.tv_factor * total_variation_loss
                    + self.pixel_factor * pixel_loss)
        else:
            return style_loss, content_loss, total_variation_loss, pixel_loss


class TotalVariationLoss(torch.nn.Module):
    def __init__(self, format_channels: str = "CWH", weight: float = 1.0):
        super().__init__()
        assert "C" in format and "W" in format and "H" in format_channels and len(format_channels) == 3, (
            "Missing element in format, expected `C` and `W` and `H`, found {}".format(format_channels)
        )

        self.weight = weight
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


class DeepEnergyLoss(torch.nn.Module):
    """
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
    """
    def __init__(self, alpha: float = 1.0, return_single: bool = True):
        super().__init__()
        self.alpha = alpha
        self.return_single = return_single

    def forward(self, pred_true, pred_false):
        loss_reg = self.alpha * (pred_true ** 2 + pred_false ** 2).mean()
        loss_div = pred_false.mean() - pred_true.mean()

        if self.return_single:
            return loss_reg + loss_div
        else:
            return loss_reg, loss_div

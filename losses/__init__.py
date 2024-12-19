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
    def __init__(self, discriminator_network: torch.nn.Module = None, features_dict: Optional[Dict[str, str]] = None, content_dict: Optional[Dict[str, str]] = None,
                 return_one: bool = True, style_factor: float = 1.0,
                 content_factor: float = 1.0, needs_norm: bool = True):
        super().__init__()
        
        self.model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.DEFAULT).features.eval()\
            if discriminator_network is None else discriminator_network

        if features_dict is None:
            self.return_features = features_dict
        else:
            self.return_features = {
                "3": "first",
                "8": "second",
                "15": "third",
                "22": "fourth"
            }

        if content_dict is None:
            self.return_content = {
                "15": "first"
            }
        else:
            self.return_content = content_dict

        self.feature_model = create_feature_extractor(self.model, self.return_features)
        self.content_model = create_feature_extractor(self.model, self.return_content)

        self.return_single = return_one
        self.content_factor = content_factor
        self.style_factor = style_factor
        self.needs_norm = needs_norm
        self.tv = TotalVariationLoss()

    def gram_matrix(self, x):
        pass

    def calc_style_loss(self, target: Dict[str, torch.Tensor], generated: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def calc_content_loss(self, original: Dict[str, torch.Tensor], generated: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def forward(self, original, target, generated):
        original_content = self.content_model(self.transforms(original) if self.needs_norm else original)
        generated_content = self.content_model(self.transforms(generated) if self.needs_norm else generated)

        generated_features = self.feature_model(self.transforms(generated) if self.needs_norm else generated)
        target_features = self.feature_model(self.transforms(target) if self.needs_norm else target)

        style_loss = 0
        content_loss = 0
        total_variation_loss = self.tv(generated)

        if self.style_factor != 0:
            style_loss = self.calc_style_loss(target_features, generated_features)

        if self.content_factor != 0:
            content_loss = self.calc_content_loss(original_content, generated_content)

        if self.return_single:
            return self.style_factor * style_loss + self.content_factor * content_loss + total_variation_loss
        else:
            return style_loss, content_loss, total_variation_loss


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

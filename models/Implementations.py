from typing import Optional, Union

import torch
import torchvision.models


class CycleGAN(torch.nn.Module):  # TODO implement replay buffer
    def __init__(self):
        super().__init__()
        self.netAB = None
        self.netBA = None
        self.discA = None
        self.discB = None

        self.idtAA = 0.0
        self.idtBB = 0.0

    def backward_gen(self, original, gen, rec, idt):
        pass

    def backward_disc(self, t, f):
        pass

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

        genA_loss = self.backward_gen(a, fakeA, recA, idtA)
        genB_loss = self.backward_gen(b, fakeB, recB, idtB)

        discA_loss = self.backward_disc(predA_true, predA_gen)
        discB_loss = self.backward_disc(predB_true, predB_gen)

        loss_AB = discA_loss + genA_loss
        loss_BA = discB_loss + genB_loss

        return loss_AB + loss_BA

    def forward(self, x):
        fakeA = self.netBA(x)
        fakeB = self.netAB(x)
        recA = self.netBA(fakeB)
        recB = self.netAB(fakeA)
        idtA = self.netBA(x)
        idtB = self.netAB(x)
        discA = self.discA(x)
        discB = self.discB(x)

        return fakeA, fakeB, recA, recB, idtA, idtB, discA, discB


class SiameseNetwork(torch.nn.Module):
    """
    https://github.com/pytorch/examples/blob/main/siamese_network/main.py
    """
    def __init__(self, backbone: Optional[torch.nn.Module], classifier: Optional[torch.nn.Module],
                 feature_union_method: str = "cat", backbone_output_shape: Optional[int] = None):
        super(SiameseNetwork, self).__init__()
        assert feature_union_method == "cat" or feature_union_method == "bilinear", (
            "Invalid union method, expected[\"cat\", \"bilinear\", \"bilinear-multi\"] " +
            "but found {}".format(feature_union_method)
        )

        if backbone is not None:
            assert backbone_output_shape is not None, "When backbone is not None, `backbone_output_shape` is expected."

        self.backbone = None
        self.classifier = None
        temp_fc_in = None

        if backbone is None:
            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            temp_fc_in = resnet.fc.in_features
            self.backbone = torch.nn.Sequential(*list(resnet.children()[:-1]))
        else:
            self.backbone = backbone
            temp_fc_in = backbone_output_shape

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
        features_x = self.backbone(x)
        features_y = self.backbone(y)

        features = None

        if self.union_method == "cat":
            features = torch.cat( (features_x, features_y), 1)
        elif self.union_method == "bilinear":  # TODO validate
            features = torch.einsum("bijk,bilm->bkm", features_x, features_y)  # batch,channel matrix mult
        elif self.union_method == "bilinear-multi":  # TODO validate
            features = torch.einsum("bijk,blmn->okn", features_x, features_y)  # all x all matrix mult

        output = self.classifier(features)

        return output

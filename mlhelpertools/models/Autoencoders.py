from typing import List, Optional

import torch


class SimpleLinearAutoencoder(torch.nn.Module):
    def __init__(self, layer_struct: List[int], activations: List[torch.nn.Module],
                 final_activation: Optional[torch.nn.Module] = None):
        super().__init__()
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()

        for i in range(1, len(layer_struct)):
            self.encoder.append(
                torch.nn.Linear(layer_struct[i - 1], layer_struct[i])
            )
            self.encoder.append(
                activations[i - 1]()
            )

            self.decoder.append(
                torch.nn.Linear(layer_struct[i], layer_struct[i - 1])
            )

            if i == len(layer_struct) - 1:
                if final_activation is not None:
                    self.decoder.append(
                        final_activation()
                    )
                else:
                    self.decoder.append(
                        activations[i]
                    )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class DynamicAutoencoder(torch.nn.Module):
    def __init__(self, blocks_enc: List[torch.nn.Module], blocks_dec: List[torch.nn.Module]):
        assert len(blocks_enc) == len(blocks_dec), "Blocks should have same size."
        super().__init__()
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()

        for i in range(len(blocks_enc)):
            self.encoder.append(
                blocks_enc[i]
            )
            self.decoder.append(
                blocks_dec[i]
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        return encoded, decoded


class DynamicVariationalAutoencoder(DynamicAutoencoder):
    def __init__(self, blocks_enc: List[torch.nn.Module], blocks_dec: List[torch.nn.Module],
                 hidden_mean: torch.nn.Module,
                 hidden_logvar: torch.nn.Module
                 ):
        super().__init__(blocks_enc, blocks_dec)
        self.hidden_mean = hidden_mean
        self.hidden_logvar = hidden_logvar

    def _reparametrization(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + logvar * eps

    def encode(self, x):
        encoded = self.encoder(x)
        m = self.hidden_mean(encoded)
        lv = self.hidden_logvar(encoded)

        return m, lv

    def decode(self, x, reparametrize=False):
        if reparametrize:
            return self.decode(self._reparametrization(*x))
        else:
            return self.decoder(x)

    def forward(self, x):
        m, lv = self.encode(x)
        decoded = self.decode(self._reparametrization(m, lv))

        return m, lv, decoded

    @staticmethod
    def loss_fn(original: torch.Tensor, generated: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor,
                repr_loss: Optional[torch.nn.Module]):
        l = repr_loss(generated, original, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return l + kl


class DynamicUNet(DynamicAutoencoder):
    def forward(self, x):
        encoded_outs = list()
        y = x

        for i in range(len(self.encoder)):
            y = self.encoder[i](y)
            encoded_outs.append(y)

        for i in range(len(self.decoder)):
            y = self.decoder[i](y + encoded_outs[len(encoded_outs) - i - 1])

        return y



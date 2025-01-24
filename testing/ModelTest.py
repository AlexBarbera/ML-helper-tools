import unittest

import matplotlib.pyplot as plt
import torch.nn
import torchvision.models.feature_extraction
from mlhelpertools import models


class SiameseNetworkTest(unittest.TestCase):
    def setUp(self):
        b = torch.nn.Sequential(
            torch.nn.Linear(3, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 10),
            torch.nn.ReLU()
        )

        c = torch.nn.Sequential(
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

        c1 = torch.nn.Sequential(
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid()
        )

        c2 = torch.nn.Sequential(
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

        self.model_linear_cat = models.SiameseNetwork(b, c, feature_union_method="cat")
        self.model_linear_diff = models.SiameseNetwork(b, c1, feature_union_method="diff")
        self.model_linear_bi = models.SiameseNetwork(b, c2, feature_union_method="bilinear")

        b = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(3, 10, 2, padding=1),
            torch.nn.ReLU()
        )

        c = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(500, 1),
            torch.nn.Sigmoid()
        )

        c1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(250, 1),
            torch.nn.Sigmoid()
        )

        c2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

        self.model_cnn_cat = models.SiameseNetwork(b, c, feature_union_method="cat")
        self.model_cnn_diff = models.SiameseNetwork(b, c1, feature_union_method="diff")
        self.model_cnn_bi = models.SiameseNetwork(b, c2, feature_union_method="bilinear")

    def test_cat_linear(self):
        data = torch.ones(1, 3)

        hidden = self.model_linear_cat.forward_backbone(data, data)
        output = self.model_linear_cat(data, data)

        self.assertEqual(hidden.ndim, 2, msg="Hidden features ndim not 2.")
        self.assertEqual(hidden.shape[0], 1, msg="Hidden features Batch size not right.")
        self.assertEqual(hidden.shape[1], 20, msg="Hidden features has wrong number of features.")

        self.assertEqual(output.ndim, 2, msg="Output has wrong ndim")

    def test_diff_linear(self):
        data = torch.ones(1, 3)

        hidden = self.model_linear_diff.forward_backbone(data, data)
        output = self.model_linear_diff(data, data)

        self.assertEqual(hidden.ndim, 2, msg="Hidden features ndim not 2.")
        self.assertEqual(hidden.shape[0], 1, msg="Hidden features Batch size not right.")
        self.assertEqual(hidden.shape[1], 10, msg="Hidden features has wrong number of features.")
        self.assertTrue(torch.equal(hidden, torch.zeros_like(hidden)))

        self.assertEqual(output.ndim, 2, msg="Output has wrong ndim")

    def test_bilinear_linear(self):
        data = torch.ones(1, 3)

        with self.assertRaises(RuntimeError, msg="Using bilinear operator on linear model does not raise error"):
            hidden = self.model_linear_bi.forward_backbone(data, data)
        with self.assertRaises(RuntimeError, msg="Using bilinear operator on linear model does not raise error"):
            output = self.model_linear_bi(data, data)

    def test_invalid_union_method(self):
        with self.assertRaises(AssertionError, msg="Invalid union method does not raise AssertionError"):
            models.SiameseNetwork(None, None, feature_union_method="c")

        with self.assertRaises(AssertionError, msg="Invalid union method does not raise AssertionError"):
            models.SiameseNetwork(None, None, feature_union_method="A")

            with self.assertRaises(AssertionError, msg="Invalid union method does not raise AssertionError"):
                models.SiameseNetwork(None, None, feature_union_method="1")

        with self.assertRaises(AssertionError, msg="Invalid union method does not raise AssertionError"):
            models.SiameseNetwork(None, None, feature_union_method=1)

        with self.assertRaises(AssertionError, msg="Invalid union method does not raise AssertionError"):
            models.SiameseNetwork(None, None, feature_union_method="BILINEAR")

    def test_cat_cnn(self):
        data = torch.ones(1, 1, 3, 3)

        hidden = self.model_cnn_cat.forward_backbone(data, data)
        output = self.model_cnn_cat(data, data)

        self.assertEqual(hidden.ndim, 4, msg="Hidden features ndim not 4.")
        self.assertEqual(hidden.shape[0], 1, msg="Hidden features Batch size not right.")
        self.assertEqual(hidden.shape[1], 20, msg="Hidden features has wrong number of features.")

        self.assertEqual(hidden.flatten(start_dim=1).shape[1], 500, "Hidden features incorrect count.")

        self.assertEqual(output.ndim, 2, msg="Output has wrong ndim")

    def test_diff_cnn(self):
        data = torch.ones(1, 1, 3, 3)

        hidden = self.model_cnn_diff.forward_backbone(data, data)
        output = self.model_cnn_diff(data, data)

        self.assertEqual(hidden.ndim, 4, msg="Hidden features ndim not 2.")
        self.assertEqual(hidden.shape[0], 1, msg="Hidden features Batch size not right.")
        self.assertEqual(hidden.shape[1], 10, msg="Hidden features has wrong number of features.")
        self.assertTrue(torch.equal(hidden, torch.zeros_like(hidden)))

        self.assertEqual(hidden.flatten(start_dim=1).shape[1], 250, "Hidden features incorrect count.")

        self.assertEqual(output.ndim, 2, msg="Output has wrong ndim")

    def test_bilinear_cnn(self):
        data = torch.ones(1, 1, 3, 3)

        hidden = self.model_cnn_bi.forward_backbone(data, data)
        output = self.model_cnn_bi(data, data)

        self.assertEqual(hidden.ndim, 3, msg="Hidden features ndim not 2.")
        self.assertEqual(hidden.shape[0], 1, msg="Hidden features Batch size not right.")
        self.assertEqual(hidden.shape[1], 10, msg="Hidden features has wrong number of features.")

        self.assertEqual(hidden.flatten(start_dim=1).shape[1], 100, "Hidden features incorrect count.")

        self.assertEqual(output.ndim, 2, msg="Output has wrong ndim")


class AutoencoderTest(unittest.TestCase):
    def test_autoencoder(self):
        enc = torch.nn.Sequential(
            torch.nn.Linear(19, 12),
            torch.nn.ReLU(),
        )

        dec = torch.nn.Sequential(
            torch.nn.Linear(12, 19),
            torch.nn.ReLU()
        )

        model = models.DynamicAutoencoder(enc, dec)
        data = torch.ones(1, 19)

        encoded1 = model.encode(data)
        decoded = model.decode(encoded1)
        encoded, reconstructed = model(data)

        self.assertTrue(torch.equal(encoded, encoded1))
        self.assertTrue(torch.equal(decoded, reconstructed))


    def test_vae(self):
        enc = torch.nn.Sequential(
            torch.nn.Linear(19, 12),
            torch.nn.ReLU(),
        )

        dec = torch.nn.Sequential(
            torch.nn.Linear(12, 19),
            torch.nn.ReLU()
        )
        m = torch.nn.Linear(12, 12)
        v = torch.nn.Linear(12, 12)
        model = models.DynamicVariationalAutoencoder(enc, dec, m, v)

        data = torch.ones(1, 19)

        encoded_mean, encoded_var = model.encode(data)
        decoded = model.decode([encoded_mean, encoded_var], reparametrize=True)
        encoded_mean1, encoded_var1, reconstructed = model(data)

        # cannot check reconstuction because reparametrization adds randomness
        self.assertTrue(torch.equal(encoded_mean, encoded_mean1))
        self.assertTrue(torch.equal(encoded_var, encoded_var1))


if __name__ == "__main__":
    dataset = torchvision.datasets.MNIST("/tmp/mnist",train=True,  download=True,
                                         transform=torchvision.transforms.Compose(
                                  [ torchvision.transforms.ToTensor(), torchvision.transforms.Resize(244),
                                    torchvision.transforms.Lambda(lambda x: x.expand(3, 244, 244))
                                    ]
                              ))

    val = torchvision.datasets.MNIST(root='/tmp/mnist_val', train=False, download=True,
                              transform=torchvision.transforms.Compose(
                                  [ torchvision.transforms.ToTensor(),torchvision.transforms.Resize(244),
                                    torchvision.transforms.Lambda(lambda x: x.expand(3, 244, 244))
                                    ]
                              )
                                     )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    val = torch.utils.data.DataLoader(val, batch_size=64, shuffle=True)
    EPOCHS = 20
    N_show = 10

    model = models.VGGAutoencoder(vgg_type="A")
    optim = torch.optim.Adam(model.parameters())

    f, ax = plt.subplots(EPOCHS, N_show)
    for epoch in range(EPOCHS):
        for x, target in train_loader:
            optim.zero_grad()
            _, y = model(x)
            loss = torch.nn.MSELoss()(x, y)
            loss.backward()
            optim.step()
        for x, target in val:
            y = model(x)
            for j in range(N_show):
                ax[epoch, j].imshow(y[j][0])
            break
        plt.draw()




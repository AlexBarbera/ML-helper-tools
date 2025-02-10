import os.path
import pickle
import unittest
import matplotlib.pyplot as plt
import torchvision.io
import torch

from mlhelpertools import utils


class MahalanobisTest(unittest.TestCase):
    def setUp(self):
        wd = os.path.dirname(__file__)
        self.a_bcwh = torchvision.io.read_image(os.path.join(wd, "cifar_6.png")).float().reshape(-1, 3, 32, 32)
        self.a_bwhc = torchvision.io.read_image(os.path.join(wd, "cifar_6.png")).float().reshape(-1, 3, 32, 32).permute(0, 2, 3, 1)
        #self.target_a_independent_bcwh = torchvision.io.read_image("cifar_6_zca_independent_channels.png").float().reshape(-1, 3, 32, 32)
        #self.target_a_no_independent_bcwh = torchvision.io.read_image("cifar_6_zca_no_independent_channels.png").float().reshape(-1, 3, 32, 32)
        #self.target_a_independent_bwhc = torchvision.io.read_image("cifar_6_zca_independent_channels.png").float().reshape(-1, 3, 32, 32).permute(0, 2, 3, 1)
        #self.target_a_no_independent_bwhc = torchvision.io.read_image("cifar_6_zca_no_independent_channels.png").float().reshape(-1, 3, 32, 32).permute(0, 2, 3, 1)

        self.target_a_independent_bcwh = None
        self.target_a_no_independent_bcwh = None
        self.target_a_independent_bwhc = None
        self.target_a_no_independent_bwhc = None

        self.target_batcha_independent_bwhc = None
        self.target_batcha_independent_bcwh = None
        self.target_batcha_no_independent_bwhc = None
        self.target_batcha_no_independent_bcwh = None

        with open(os.path.join(wd, "cifar_6_zca_independent_channels.pkl"), "rb") as f:  # 32,32,3
            self.target_a_independent_bwhc = pickle.load(f)
            self.target_a_independent_bwhc = self.target_a_independent_bwhc.reshape(1, *self.target_a_independent_bwhc.shape)
        with open(os.path.join(wd, "cifar_6_zca_independent_channels.pkl"), "rb") as f:  # 32,32,3
            self.target_a_independent_bcwh = pickle.load(f).permute(2, 0, 1)
            self.target_a_independent_bcwh = self.target_a_independent_bcwh.reshape(1, *self.target_a_independent_bcwh.shape)

        with open(os.path.join(wd, "cifar_6_zca_no_independent_channels.pkl"), "rb") as f:  # 32,32,3
            self.target_a_no_independent_bwhc = pickle.load(f)
            self.target_a_no_independent_bwhc = self.target_a_no_independent_bwhc.reshape(1, *self.target_a_no_independent_bwhc.shape)
        with open(os.path.join(wd, "cifar_6_zca_no_independent_channels.pkl"), "rb") as f:  # 32,32,3
            self.target_a_no_independent_bcwh = pickle.load(f).permute(2, 0, 1)
            self.target_a_no_independent_bcwh = self.target_a_no_independent_bcwh.reshape(1, *self.target_a_no_independent_bcwh.shape)

        with open(os.path.join(wd, "cifar_batch2x6_zca_independent_channels.pkl"), "rb") as f:
            self.target_batcha_independent_bcwh = pickle.load(f).permute(0, 3, 1, 2)
        with open(os.path.join(wd, "cifar_batch2x6_zca_independent_channels.pkl"), "rb") as f:
            self.target_batcha_independent_bwhc = pickle.load(f)

        with open(os.path.join(wd, "cifar_batch2x6_zca_no_independent_channels.pkl"), "rb") as f:
            self.target_batcha_no_independent_bcwh = pickle.load(f).permute(0, 3, 1, 2)
        with open(os.path.join(wd, "cifar_batch2x6_zca_no_independent_channels.pkl"), "rb") as f:
            self.target_batcha_no_independent_bwhc = pickle.load(f)

    def test_independent_single_bcwh(self):
        x = utils.mahalanobis_whitening_batch(self.a_bcwh, batch_format="BCWH", independent_channels=True)

        self.assertTrue(torch.equal(x, self.target_a_independent_bcwh),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_independent_single_bwhc(self):
        x = utils.mahalanobis_whitening_batch(self.a_bwhc, batch_format="BWHC", independent_channels=True)

        self.assertTrue(torch.equal(x, self.target_a_independent_bwhc),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_not_independent_single_bcwh(self):
        x = utils.mahalanobis_whitening_batch(self.a_bcwh, batch_format="BCWH", independent_channels=False)

        self.assertTrue(torch.equal(x, self.target_a_no_independent_bcwh),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_not_independent_single_bwhc(self):
        x = utils.mahalanobis_whitening_batch(self.a_bwhc, batch_format="BWHC", independent_channels=False)

        self.assertTrue(torch.equal(x, self.target_a_no_independent_bwhc),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_independent_batch_bcwh(self):
        b1 = torch.vstack([self.a_bcwh, self.a_bcwh])

        x = utils.mahalanobis_whitening_batch(b1, batch_format="BCWH", independent_channels=True)

        #self.assertTrue(torch.equal(x[0], x[1]))

        self.assertTrue(torch.equal(x, self.target_batcha_independent_bcwh),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_independent_batch_bwhc(self):
        b1 = torch.vstack([self.a_bwhc, self.a_bwhc])

        x = utils.mahalanobis_whitening_batch(b1, batch_format="BWHC", independent_channels=True)

        self.assertTrue(torch.equal(x, self.target_batcha_independent_bwhc),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_not_independent_batch_bcwh(self):
        b1 = torch.vstack([self.a_bcwh, self.a_bcwh])

        x = utils.mahalanobis_whitening_batch(b1, batch_format="BCWH", independent_channels=False)

        self.assertTrue(torch.equal(x, self.target_batcha_no_independent_bcwh),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_not_independent_batch_bwhc(self):
        b1 = torch.vstack([self.a_bwhc, self.a_bwhc])

        x = utils.mahalanobis_whitening_batch(b1, batch_format="BWHC", independent_channels=False)

        self.assertTrue(torch.equal(x, self.target_batcha_no_independent_bwhc),
                         msg="Target for ZCA whitening for independent channels not correct.")

    def test_tonbits(self):
        x = (torch.ones(10) * (((2**16)-1)/2)).type(torch.int16)
        expected = (torch.ones(10) * 255//2).type(torch.uint8)
        res = utils.to_nbit(x, torch.uint8)
        self.assertTrue(
            torch.allclose(res, expected),
            msg="Wrong values:\n{}\n{}".format(expected, res)
        )

        x = (torch.ones(10) * ((2 ** 16) - 1)).type(torch.uint16)
        expected = (torch.ones(10) * 255).type(torch.uint8)
        res = utils.to_nbit(x, torch.uint8)
        self.assertTrue(
            torch.allclose(res, expected),
            msg="Wrong values:\n{}\n{}".format(expected, res)
        )

        x = (torch.zeros(2, 10)).type(torch.int16)
        expected = (torch.zeros(2, 10)).type(torch.uint8)
        res = utils.to_nbit(x, torch.uint8)
        self.assertTrue(
            torch.allclose(res, expected),
            msg="Wrong values:\n{}\n{}".format(expected, res)
        )

        x = (torch.ones(10) * 255).type(torch.uint8)
        expected = (torch.ones(10) * (2**16 - 1)).type(torch.uint16)
        res = utils.to_nbit(x, torch.uint16)
        self.assertTrue(
            torch.allclose(res, expected),
            msg="Wrong values:\n{}\n{}".format(expected, res)
        )

    def test_l2norm(self):
        x = torch.tensor(range(12)).reshape((1, 3, 4))

        expected = torch.tensor(22.4944)
        res = utils.l2_norm(x)
        self.assertTrue(
            torch.isclose(expected, res),
            msg="Values not equal.\n{}\n{}".format(expected, res)
        )

        x = torch.tensor(range(12)).reshape((1, 12))
        res = utils.l2_norm(x)
        self.assertTrue(
            torch.isclose(expected, res),
            msg="Values not equal.\n{}\n{}".format(expected, res)
        )

        x = torch.zeros(1, 3, 4)

        expected = torch.tensor(0).float()
        res = utils.l2_norm(x)
        self.assertTrue(
            torch.isclose(expected, res),
            msg="Values not equal.\n{}\n{}".format(expected, res)
        )

    def test_minmaxscale(self):
        x = torch.tensor(range(12)).reshape(1, 3, 4).float()
        expected = (x - 0) / 11

        res = utils.minmax_scale(x)

        self.assertTrue(
            torch.equal(expected, res),
            msg="Values dont match.\n{}\n{}".format(expected, res)
        )

    def test_zscore(self):
        data = torch.tensor(range(10))
        target = (torch.tensor(range(10)) - 4.5) / 3.027
        output, mean, std = utils.zscore(data)

        self.assertEqual(mean.item(), 4.5, msg="Incorrect mean.")
        self.assertAlmostEqual(std.item(), 3.027, places=2, msg="Incorrect var.")
        self.assertTrue(
            torch.allclose(output, target, rtol=1e-3, atol=1e-3), msg="Incorrect normalized zscore tensor."
        )

        data = data.reshape((2, 5))
        target = target.reshape((2, 5))

        output2, mean2, std2 = utils.zscore(data)

        self.assertEqual(mean, mean2, msg="Means not equal when shape is different.")
        self.assertEqual(std, std2, msg="Var not equal when shape is different.")
        self.assertTrue(
            torch.allclose(output2, target, rtol=1e-3, atol=1e-3), msg="Incorrect normalized zscore tensor (2-D)."
        )


def zca_imshow(ax, image):
    m, M = image.min(), image.max()
    ax.imshow((image - m) / (M - m))

def norm(x):
    m, M = x.min(), x.max()
    return (x - m) / (M - m)

def test_normal():
    x = torchvision.io.read_image("testing/cifar_6.png").reshape(-1, 3, 32, 32)
    x = x.permute(0, 2, 3, 1)  # B W H C

    xp1 = utils.mahalanobis_whitening_batch(x.clone(), batch_format="BWHC", independent_channels=True)
    xp2 = utils.mahalanobis_whitening_batch(x.clone(), batch_format="BWHC", independent_channels=False)

    fig, ax = plt.subplots(1, 3)
    ax = ax.flatten()

    print(xp1.shape)

    ax[0].imshow(x[0])
    zca_imshow(ax[1], xp1[0])
    zca_imshow(ax[2], xp2[0])

    print(torch.equal(xp1[0], xp1[1]),
          torch.equal(norm(xp1[0]), norm(xp1[1]))
          )

    """
    with open("testing/cifar_6_zca_independent_channels.pkl", "wb") as f:
        pickle.dump(xp1[0], f)

    with open("testing/cifar_6_zca_no_independent_channels.pkl", "wb") as f:
        pickle.dump(xp2[0], f)
    """
    """
    m, M = xp1[0].min(), xp1[0].max()
    xp1 = (xp1[0] - m) / (M - m)

    m, M = xp2[0].min(), xp2[0].max()
    xp2 = (xp2[0] - m) / (M - m)

    torchvision.utils.save_image(xp1.permute(2,0,1), "testing/cifar_6_zca_independent_channels.png")
    torchvision.utils.save_image(xp2.permute(2,0,1), "testing/cifar_6_zca_no_independent_channels.png")
    """
    plt.show()


def debug():
    x = torchvision.io.read_image("testing/cifar_6.png").reshape(-1, 3, 32, 32)
    x = x.permute(0, 2, 3, 1)  # B W H C

    xp1 = utils.mahalanobis_whitening_batch(x.clone(), batch_format="BWHC", independent_channels=True)
    xp2 = utils.mahalanobis_whitening_batch(x.clone(), batch_format="BWHC", independent_channels=False)

    fig, ax = plt.subplots(3, 3)
    ax = ax.flatten()

    print(xp1.shape, xp2.shape)

    ax[0].imshow(x[0])
    ax[3].imshow(x[0])
    ax[6].imshow(x[0])

    zca_imshow(ax[1], xp1[0])
    zca_imshow(ax[2], xp2[0])

    b = torch.vstack([x, x])

    xp1 = utils.mahalanobis_whitening_batch(b.clone(), batch_format="BWHC", independent_channels=True)
    xp2 = utils.mahalanobis_whitening_batch(b.clone(), batch_format="BWHC", independent_channels=False)

    print(xp1.shape)

    zca_imshow(ax[4], xp1[0])
    zca_imshow(ax[5], xp1[1])
    zca_imshow(ax[7], xp2[0])
    zca_imshow(ax[8], xp2[1])

    plt.show()

    with open("testing/cifar_batch2x6_zca_independent_channels.pkl", "wb") as f:
        pickle.dump(xp1, f)

    with open("testing/cifar_batch2x6_zca_no_independent_channels.pkl", "wb") as f:
        pickle.dump(xp2, f)


if __name__ == "__main__":
    debug()

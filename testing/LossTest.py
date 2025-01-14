import unittest
import torch
import torchvision.io

import losses


class TotalVariationLossTest(unittest.TestCase):
    def setUp(self):
        self.loss = losses.TotalVariationLoss()
        self.delta = 0.001
        torch.manual_seed(1)

    def test_zero(self):
        original = torch.ones(1, 1, 10, 10)

        self.assertAlmostEquals(self.loss(original), 0, delta=self.delta)
        self.assertAlmostEquals(self.loss(original * 0), 0, delta=self.delta)
        self.assertAlmostEquals(self.loss(original * -1), 0, delta=self.delta)

    def test_notzero(self):
        original = torch.tensor([[[[0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
          [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
          [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
          [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
          [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]])

        expected = 0.1369

        self.assertAlmostEquals(self.loss(original), expected, delta=self.delta)
        self.assertAlmostEquals(self.loss(original * 2), expected * 4, delta=self.delta)
        self.assertAlmostEquals(self.loss(original * -1), expected, delta=self.delta)


class PerceptualLossTest(unittest.TestCase):
    def setUp(self):
        self.loss = losses.PerceptualLoss(tv_factor=0)
        self.data1 = torchvision.io.read_image("cifar_6.png").float().reshape(-1, 3, 32, 32)
        self.data2 = torchvision.io.read_image("cifar_9.png").float().reshape(-1, 3, 32, 32)

    def test_equal(self):
        self.assertEqual(
            self.loss(self.data1, self.data1, self.data1), 0
        )

    def test_notEqual(self):
        self.assertNotEquals(
            self.loss(self.data1, self.data1, self.data2), 0
        )


class DeepEneryLossTest(unittest.TestCase):
    def setUp(self):
        self.loss = losses.DeepEnergyLoss(reduction=False)
        self.loss2 = losses.DeepEnergyLoss(alpha=0.5)

    def test_zero(self):
        reg, div = self.loss(torch.ones(3), torch.zeros(3))

        self.assertEqual(reg, 1)
        self.assertEqual(div, -1)
        self.assertEqual(self.loss2(torch.ones(3), torch.zeros(3)), -0.5)

    def test_nonzero(self):
        a = torch.tensor([0.1, 0.2, 0.3])
        b = torch.tensor([0.9, 0.8, 0.7])
        reg, div = self.loss(a, b)

        self.assertAlmostEqual(reg.item(), 0.6933, places=4)
        self.assertAlmostEqual(div, 0.6, places=2)
        self.assertAlmostEqual(self.loss2(a, b).item(), 0.9467, places=4)
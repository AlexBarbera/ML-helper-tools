import unittest
import torch
import torchvision.io
import os

from mlhelpertools import losses


class TotalVariationLossTest(unittest.TestCase):
    def setUp(self):
        self.loss = losses.TotalVariationLoss()
        self.loss1 = losses.TotalVariationLoss(format_channels="WCH")
        self.loss2 = losses.TotalVariationLoss(format_channels="WHC")
        self.delta = 0.001
        torch.manual_seed(1)

    def test_zero(self):
        original = torch.ones(1, 1, 10, 10)

        self.assertAlmostEqual(self.loss(original), 0, delta=self.delta)
        self.assertAlmostEqual(self.loss(original * 0), 0, delta=self.delta)
        self.assertAlmostEqual(self.loss(original * -1), 0, delta=self.delta)

        original_permuted = original.permute((0, 2, 1, 3))  # BWCH

        self.assertAlmostEqual(self.loss1(original_permuted), 0, delta=self.delta)
        self.assertAlmostEqual(self.loss1(original_permuted * 2), 0, delta=self.delta)
        self.assertAlmostEqual(self.loss1(original_permuted * -1), 0, delta=self.delta)

        original_permuted = original.permute((0, 2, 3, 1))  # BWHC

        self.assertAlmostEqual(self.loss2(original_permuted), 0, delta=self.delta)
        self.assertAlmostEqual(self.loss2(original_permuted * 2), 0, delta=self.delta)
        self.assertAlmostEqual(self.loss2(original_permuted * -1), 0, delta=self.delta)

        self.assertTrue(
            torch.equal(self.loss(original), self.loss2(original_permuted))
        )

    def test_notzero(self):
        original = torch.tensor([[[[0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
          [0.7999, 0.3971, 0.7544, 0.5695, 0.4388],
          [0.6387, 0.5247, 0.6826, 0.3051, 0.4635],
          [0.4550, 0.5725, 0.4980, 0.9371, 0.6556],
          [0.3138, 0.1980, 0.4162, 0.2843, 0.3398]]]])

        expected = 0.1369

        self.assertAlmostEqual(self.loss(original), expected, delta=self.delta)
        self.assertAlmostEqual(self.loss(original * 2), expected * 4, delta=self.delta)
        self.assertAlmostEqual(self.loss(original * -1), expected, delta=self.delta)

        original_permuted = original.permute((0, 2, 1, 3))  # BWCH

        self.assertAlmostEqual(self.loss1(original_permuted), expected, delta=self.delta)
        self.assertAlmostEqual(self.loss1(original_permuted * 2), expected * 4, delta=self.delta)
        self.assertAlmostEqual(self.loss1(original_permuted * -1), expected, delta=self.delta)

        self.assertTrue(
            torch.equal(self.loss(original), self.loss1(original_permuted))
        )

        original_permuted = original.permute((0, 2, 3, 1))  # BWHC

        self.assertAlmostEqual(self.loss2(original_permuted), expected, delta=self.delta)
        self.assertAlmostEqual(self.loss2(original_permuted * 2), expected * 4, delta=self.delta)
        self.assertAlmostEqual(self.loss2(original_permuted * -1), expected, delta=self.delta)

        self.assertTrue(
            torch.equal(self.loss(original), self.loss2(original_permuted))
        )

    def testInvalidChannelFormat(self):
        with self.assertRaises(AssertionError):
            losses.TotalVariationLoss(format_channels="WCHB")
        with self.assertRaises(AssertionError):
            losses.TotalVariationLoss(format_channels="WACH")
        with self.assertRaises(AssertionError):
            losses.TotalVariationLoss(format_channels="1")


class PerceptualLossTest(unittest.TestCase):
    def setUp(self):
        wd = os.path.dirname(__file__)
        self.loss = losses.PerceptualLoss(tv_factor=0)
        self.loss1 = losses.PerceptualLoss(pixel_factor=1.0, reduction=False)

        self.data1 = torchvision.io.read_image(os.path.join(wd, "cifar_6.png")).float().reshape(-1, 3, 32, 32)
        self.data2 = torchvision.io.read_image(os.path.join(wd, "cifar_9.png")).float().reshape(-1, 3, 32, 32)

    def test_equal(self):
        self.assertEqual(
            self.loss(self.data1, self.data1, self.data1), 0
        )

    def test_notEqual(self):
        self.assertNotEqual(
            self.loss(self.data1, self.data1, self.data2), 0
        )

    def test_noReduction(self):
        l = self.loss1(self.data1, self.data1, self.data1)

        self.assertEqual(len(l), 4, "Imcorrect number of output losses")
        self.assertEqual(l[0].item(), 0, "Style loss not 0.")
        self.assertEqual(l[1].item(), 0, "Content loss not 0.")
        self.assertNotEqual(l[2].item(), 0, "TV loss not correct value.")
        self.assertEqual(l[3].item(), 0, "Pixel loss not 0.")


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


class WassersteinLossTest(unittest.TestCase):
    def setUp(self):
        self.loss = losses.WassersteinLoss(format_channels="CWH")
        self.loss1 = losses.WassersteinLoss(format_channels="WHC")
        self.loss2 = losses.WassersteinLoss(format_channels="WCH")

    def test_Wasserstein_init(self):
        with self.assertRaises(AssertionError, msg="Invalid channel format should raise error"):
            l = losses.WassersteinLoss(format_channels="CAH")
        with self.assertRaises(AssertionError, msg="Invalid channel format should raise error"):
            l = losses.WassersteinLoss(format_channels="CWA")
        with self.assertRaises(AssertionError, msg="Invalid channel format should raise error"):
            l = losses.WassersteinLoss(format_channels="AWH")

    def test_equal(self):
        data = torch.ones(1, 24, 24)
        data = torch.randn_like(data)
        l = self.loss(data, data)

        self.assertTrue(torch.equal(l, torch.tensor(0)))

        data_permuted = data.permute((1, 2, 0))  # WHC
        l1 = self.loss1(data_permuted, data_permuted)
        self.assertTrue(torch.equal(l1, torch.tensor(0)))
        self.assertTrue(torch.equal(l, l1))

        data_permuted = data.permute((1, 0, 2))  # WCH
        l1 = self.loss1(data_permuted, data_permuted)
        self.assertTrue(torch.equal(l1, torch.tensor(0)))
        self.assertTrue(torch.equal(l, l1))

    def test_equal_batched(self):
        data = torch.ones(2, 1, 24, 24)
        data = torch.randn_like(data)
        l = self.loss(data, data)

        self.assertTrue(torch.equal(l, torch.tensor(0)))

        data_permuted = data.permute((0, 2, 3, 1))  # WHC
        l1 = self.loss1(data_permuted, data_permuted)
        self.assertTrue(torch.equal(l1, torch.tensor(0)))
        self.assertTrue(torch.equal(l, l1))

        data_permuted = data.permute((0, 2, 1, 3))  # WCH
        l1 = self.loss2(data_permuted, data_permuted)
        self.assertTrue(torch.equal(l1, torch.tensor(0)))
        self.assertTrue(torch.equal(l, l1))

    def test_pointlcloud_output(self):
        B = 2
        C = 4
        W = 24
        H = 13

        data = torch.ones(B, C, W, H)
        data = torch.randn_like(data)
        self.assertEqual(self.loss.to_pointcloud(data).shape, (B, W * H, C))

        data = torch.ones(B, W, C, H)
        data = torch.randn_like(data)
        self.assertEqual(self.loss2.to_pointcloud(data).shape, (B, W * H, C))

        data = torch.ones(B, W, H, C)
        data = torch.randn_like(data)
        self.assertEqual(self.loss1.to_pointcloud(data).shape, (B, W * H, C))

        # Unbatched
        data = torch.ones(C, W, H)
        data = torch.randn_like(data)
        self.assertEqual(self.loss.to_pointcloud(data).shape, (1, W * H, C))

        data = torch.ones(W, C, H)
        data = torch.randn_like(data)
        self.assertEqual(self.loss2.to_pointcloud(data).shape, (1, W * H, C))

        data = torch.ones(W, H, C)
        data = torch.randn_like(data)
        self.assertEqual(self.loss1.to_pointcloud(data).shape, (1, W * H, C))

    def test_diff(self):
        data = torch.ones(1, 24, 24)
        data = torch.randn_like(data)
        data1 = torch.randn_like(data)
        l = self.loss(data, data1)

        self.assertFalse(torch.equal(l, torch.tensor(0)))

        data_permuted = data.permute((1, 2, 0))  # WHC
        data_permuted1 = data1.permute((1, 2, 0))
        l1 = self.loss1(data_permuted, data_permuted1)
        self.assertFalse(torch.equal(l1, torch.tensor(0)))
        self.assertTrue(torch.equal(l, l1))

        data_permuted = data.permute((1, 0, 2))  # WCH
        data_permuted1 = data1.permute((1, 0, 2))
        l1 = self.loss2(data_permuted, data_permuted1)
        self.assertFalse(torch.equal(l1, torch.tensor(0)))
        self.assertTrue(torch.equal(l, l1))


class WordTreeTransformTest(unittest.TestCase):
    def setUp(self):
        self.words = [
            "pet", "wild",
            "cat", "dog", "fish",
            "scorpion", "croc",
            "tubby", "persian",
            "german shepard", "husky", "malamute",
            "goldfish", "not goldfish",  # here is where i run out of imagination
            "poison boi", "pincers boi",
            "crocodile", "aligator"
        ]

        self.relations = dict(
            pet=["cat", "dog", "fish"],
            wild=["scorpion", "croc"],
            cat=["tubby", 'persian'],
            dog=["german shepard", "husky", "malamute"],
            fish=["goldfish", "not goldfish"],
            scorpion=["poison boi", "pincers boi"],
            croc=["crocodile", "aligator"]
        )
        self.wt = losses.WordTreeTransformation(words=self.words, relations=self.relations)

    def testWordTreeDepth0ByStr(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("pet")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet not selected for \"pet\".\n{}\n{}".format(expected, predicted)
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("wild")

        self.assertTrue(
            torch.equal(expected, predicted), "wild not selected for \"wild\".\n{}\n{}".format(expected, predicted)
        )

    def testWordTreeDepth0ByInt(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("pet"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet not selected for \"{}\".\n{}\n{}".format(self.words.index("pet"), expected, predicted)
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("wild"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Wild not selected for \"{}\".\n{}\n{}".format(self.words.index("wild"), expected, predicted)
        )

    def testWordTreeDepth1ByStr(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("cat")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|cat not selected for \"cat\".\n{}\n{}".format(expected, predicted)
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("dog")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|dog not selected for \"dog\".\n{}\n{}".format(expected, predicted)
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("fish")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("fish")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|fish not selected for \"fish\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("scorpion")

        self.assertTrue(
            torch.equal(expected, predicted), "Wild|scorpion not selected for \"scorpion\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("croc")

        self.assertTrue(
            torch.equal(expected, predicted), "Wild|croc not selected for \"croc\""
        )

    def testWordTreeDepth1ByInt(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("cat"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|cat not selected for \"{}\"".format(self.words.index("cat"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("dog"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|dog not selected for \"{}\"".format(self.words.index("dog"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("fish")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("fish"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|cat not selected for \"{}\"".format(self.words.index("fish"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("scorpion"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Wild|scorpion not selected for \"{}\"".format(self.words.index("scorpion"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("croc"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Wild|croc not selected for \"{}\"".format(self.words.index("croc"))
        )

    def testWordTreeDepth2ByStr(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("tubby")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("tubby")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|cat|tubby not selected for \"tubby\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("persian")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("persian")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|cat|persian not selected for \"persian\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("german shepard")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("german shepard")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|dog|german shepard not selected for \"german shepard\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("husky")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("husky")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|dog|husky not selected for \"husky\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("malamute")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("malamute")

        self.assertTrue(
            torch.equal(expected, predicted), "Pet|dog|malamute not selected for \"malamute\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("poison boi")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("poison boi")

        self.assertTrue(
            torch.equal(expected, predicted), "Wild|scorpion|poison boi not selected for \"poison boi\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("pincers boi")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("pincers boi")

        self.assertTrue(
            torch.equal(expected, predicted), "Wild|scorpion|pincers boi not selected for \"pincers boi\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("crocodile")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("crocodile")

        self.assertTrue(
            torch.equal(expected, predicted), "Wild|croc|crocodile not selected for \"crocodile\""
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("aligator")] = 1.0
        predicted = self.wt.logit_to_flattened_tree("aligator")

        self.assertTrue(
            torch.equal(expected, predicted), "Wild|croc|aligator not selected for \"aligator\""
        )

    def testWordTreeDepth2ByInt(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("tubby")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("tubby"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|cat|tubby not selected for \"{}\"".format(self.words.index("tubby"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("persian")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("persian"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|cat|persian not selected for \"{}\"".format(self.words.index("persian"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("german shepard")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("german shepard"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|dog|german shepard not selected for \"{}\"".format(self.words.index("german shepard"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("husky")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("husky"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|dog|husky not selected for \"{}\"".format(self.words.index("husky"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("malamute")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("malamute"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Pet|dog|malamute not selected for \"{}\"".format(self.words.index("malamute"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("poison boi")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("poison boi"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Wild|scorpion|poison boi not selected for \"{}\"".format(self.words.index("poison boi"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("pincers boi")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("pincers boi"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Wild|scorpion|pincers boi not selected for \"{}\"".format(self.words.index("pincers boi"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("crocodile")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("crocodile"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Wild|croc|crocodile not selected for \"{}\"".format(self.words.index("crocodile"))
        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("aligator")] = 1.0
        predicted = self.wt.logit_to_flattened_tree(self.words.index("aligator"))

        self.assertTrue(
            torch.equal(expected, predicted),
            "Wild|croc|aligator not selected for \"{}\"".format(self.words.index("aligator"))
        )

    def testWordTreeBatch(self):
        expected = torch.zeros(3, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0

        expected[1, self.words.index("pet")] = 1.0
        expected[1, self.words.index("dog")] = 1.0
        expected[1, self.words.index("husky")] = 1.0

        expected[2, self.words.index("wild")] = 1.0
        expected[2, self.words.index("scorpion")] = 1.0
        expected[2, self.words.index("poison boi")] = 1.0

        predicted = self.wt.logit_to_flattened_tree(["cat", "husky", "poison boi"])

        self.assertTrue(
            torch.equal(predicted, expected), "Batched prediction not the same for list of strings"
        )

        predicted = self.wt.logit_to_flattened_tree([
            self.words.index("cat"),
            self.words.index("husky"),
            self.words.index("poison boi")
        ])

        self.assertTrue(
            torch.equal(predicted, expected), "Batched prediction not the same for list of ints"
        )

    def testInvalidTag(self):
        with self.assertRaises(IndexError):
            self.wt.logit_to_flattened_tree("house")

        with self.assertRaises(IndexError):
            self.wt.logit_to_flattened_tree(123456789)

        with self.assertRaises(IndexError):
            self.wt.logit_to_flattened_tree(-1)

    def testInvalidIndex(self):
        words = ["a", "b", "c"]
        relations = dict(a=["b"])

        with self.assertRaises(AssertionError):
            losses.WordTreeTransformation(words=words, relations=relations)

        words = ["a", "b", "c"]
        relations = dict(a=["b"], b=["c"], c=["a"], d=["a"])

        with self.assertRaises(AssertionError):
            losses.WordTreeTransformation(words=words, relations=relations)

    def testYieldChildrenFromPrediction(self):
        pred = list(self.wt.yield_children_from_prediction("dog"))
        expected = [[0, 1], [2, 3, 4]]

        self.assertEqual(pred, expected, msg="Invalid levels.")

        pred = list(self.wt.yield_children_from_prediction("cat"))
        expected = [[0, 1], [2, 3, 4]]

        self.assertEqual(pred, expected, msg="Invalid levels.")

        pred = list(self.wt.yield_children_from_prediction("wild"))
        expected = [[0, 1]]

        self.assertEqual(pred, expected, msg="Invalid levels.")

        pred = list(self.wt.yield_children_from_prediction("crocodile"))
        expected = [[0, 1], [5, 6], [16, 17]]

        self.assertEqual(pred, expected, msg="Invalid levels.")

        pred = list(self.wt.yield_children_from_prediction("aligator"))
        expected = [[0, 1], [5, 6], [16, 17]]

        self.assertEqual(pred, expected, msg="Invalid levels.")

        with self.assertRaises(IndexError):
            x = list(self.wt.yield_children_from_prediction("abc"))



class WordTreeLossTest(unittest.TestCase):
    def setUp(self):
        self.words = [
            "pet", "wild",
            "cat", "dog", "fish",
            "scorpion", "croc",
            "tubby", "persian",
            "german shepard", "husky", "malamute",
            "goldfish", "not goldfish",  # here is where i run out of imagination
            "poison boi", "pincers boi",
            "crocodile", "aligator"
        ]

        self.relations = dict(
            pet=["cat", "dog", "fish"],
            wild=["scorpion", "croc"],
            cat=["tubby", 'persian'],
            dog=["german shepard", "husky", "malamute"],
            fish=["goldfish", "not goldfish"],
            scorpion=["poison boi", "pincers boi"],
            croc=["crocodile", "aligator"]
        )
        self.wt = losses.WordTreeTransformation(words=self.words, relations=self.relations)

        self.loss = losses.WordTreeLoss(self.wt)

    def testWordTreeLossDepth0ByStr(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0

        res = self.loss(expected, "pet")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0

        res = self.loss(expected, "wild")

        self.assertEqual(res.item(), 0)

        self.assertNotEqual(self.loss(expected, "dog"), 0)
        self.assertNotEqual(self.loss(expected, "cat"), 0)
        self.assertNotEqual(self.loss(expected, "fish"), 0)

    def testWordTreeDepth0ByInt(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0

        res = self.loss(expected, 0)

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0

        res = self.loss(expected, 1)

        self.assertEqual(res.item(), 0)

    def testWordTreeDepth1ByStr(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0

        res = self.loss(expected, "cat")

        self.assertEqual(res.item(), 0)

        self.assertTrue(self.loss(expected, "scorpion") > self.loss(expected, "dog"),
                        "When expected \'{}\', predicted \'{}\' (incorrect subclass) should have better "
                        "score than \'{}\' (incorrect higher tree).".format("cat", "dog", "scorpion"))

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0

        res = self.loss(expected, "dog")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("fish")] = 1.0

        res = self.loss(expected, "fish")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0

        res = self.loss(expected, "scorpion")

        self.assertEqual(res.item(), 0)

        self.assertTrue(self.loss(expected, "fish") > self.loss(expected, "croc"),
                        "When expected \'{}\', predicted \'{}\' (incorrect subclass) should have better "
                        "score than \'{}\' (incorrect higher tree).".format("scorpion", "fish", "croc"))

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        res = self.loss(expected, "croc")

        self.assertEqual(res.item(), 0)

        self.assertTrue(self.loss(expected, "cat") > self.loss(expected, "scorpion"),
                        "When expected \'{}\', predicted \'{}\' (incorrect subclass) should have better "
                        "score than \'{}\' (incorrect higher tree).".format("croc", "cat", "scorpion"))

    def testWordTreeDepth1ByInt(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0

        res = self.loss(expected, self.words.index("cat"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        res = self.loss(expected, self.words.index("dog"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("fish")] = 1.0
        res = self.loss(expected, self.words.index("fish"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0

        res = self.loss(expected, self.words.index("scorpion"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0

        res = self.loss(expected, self.words.index("croc"))

        self.assertEqual(res.item(), 0)

    def testWordTreeDepth2ByStr(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("tubby")] = 1.0
        res = self.loss(expected, "tubby")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("persian")] = 1.0
        res = self.loss(expected, "persian")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("german shepard")] = 1.0
        res = self.loss(expected, "german shepard")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("husky")] = 1.0
        res = self.loss(expected, "husky")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("malamute")] = 1.0
        res = self.loss(expected, "malamute")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("poison boi")] = 1.0
        res = self.loss(expected, "poison boi")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("pincers boi")] = 1.0
        res = self.loss(expected, "pincers boi")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("crocodile")] = 1.0
        res = self.loss(expected, "crocodile")

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("aligator")] = 1.0
        res = self.loss(expected, "aligator")

        self.assertEqual(res.item(), 0)

    def testWordTreeDepth2ByInt(self):
        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("tubby")] = 1.0
        res = self.loss(expected, self.words.index("tubby"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0
        expected[0, self.words.index("persian")] = 1.0
        res = self.loss(expected, self.words.index("persian"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("german shepard")] = 1.0
        res = self.loss(expected, self.words.index("german shepard"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("husky")] = 1.0
        res = self.loss(expected, self.words.index("husky"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("dog")] = 1.0
        expected[0, self.words.index("malamute")] = 1.0
        res = self.loss(expected, self.words.index("malamute"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("poison boi")] = 1.0
        res = self.loss(expected, self.words.index("poison boi"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("scorpion")] = 1.0
        expected[0, self.words.index("pincers boi")] = 1.0
        res = self.loss(expected, self.words.index("pincers boi"))

        self.assertEqual(res.item(), 0)

        self.assertTrue(self.loss(expected, "poison boi") < self.loss(expected, "cat"),
                        "When expected \'{}\', predicted \'{}\' (incorrect subclass) should have better "
                        "score than \'{}\' (incorrect higher tree).".format("pincers boi", "poison boi", "cat")
                        )

        self.assertTrue(self.loss(expected, "scorpion") < self.loss(expected, "croc"),
                        "When expected \'{}\', predicted \'{}\' (incorrect subclass) should have better "
                        "score than \'{}\' (incorrect higher tree).".format("pincers boi", "scorpion", "croc")
                        )

        self.assertTrue(self.loss(expected, "croc") < self.loss(expected, "cat"),
                        "When expected \'{}\', predicted \'{}\' (incorrect subclass) should have better "
                        "score than \'{}\' (incorrect higher tree).".format("pincers boi", "croc", "cat")
                        )

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("crocodile")] = 1.0
        res = self.loss(expected, self.words.index("crocodile"))

        self.assertEqual(res.item(), 0)

        expected = torch.zeros(1, len(self.words)).float()
        expected[0, self.words.index("wild")] = 1.0
        expected[0, self.words.index("croc")] = 1.0
        expected[0, self.words.index("aligator")] = 1.0
        res = self.loss(expected, self.words.index("aligator"))

        self.assertEqual(res.item(), 0)

    def testWordTreeBatch(self):
        expected = torch.zeros(3, len(self.words)).float()
        expected[0, self.words.index("pet")] = 1.0
        expected[0, self.words.index("cat")] = 1.0

        expected[1, self.words.index("pet")] = 1.0
        expected[1, self.words.index("dog")] = 1.0
        expected[1, self.words.index("husky")] = 1.0

        expected[2, self.words.index("wild")] = 1.0
        expected[2, self.words.index("scorpion")] = 1.0
        expected[2, self.words.index("poison boi")] = 1.0

        res = self.loss(expected, ["cat", "husky", "poison boi"])

        self.assertTrue(
            torch.equal(res, torch.zeros(3, 1))
        )

        res = self.loss(expected, [
            self.words.index("cat"),
            self.words.index("husky"),
            self.words.index("poison boi")
        ])

        self.assertTrue(
            torch.equal(res, torch.zeros(3, 1))
        )


class LabelSmoothingTest(unittest.TestCase):
    def setUp(self):
        self.factor = 0.9
        self.loss = losses.BinaryLabelSmoothingLoss(self.factor)
        self.loss_one_sided = losses.BinaryLabelSmoothingLoss(self.factor, one_sided=True)

    def testSmoothing(self):
        factor = 0.1
        smear = factor / (10 - 1)
        base = torch.zeros(1, 10).float()
        base[0, 3] = 1.0

        expected = torch.ones(1, 10).float() * smear
        expected[0, 3] = 1.0 - factor

        loss = losses.BinaryLabelSmoothingLoss(1.0 - factor)

        self.assertTrue(
            torch.equal(loss._smooth(base), expected)
        )

        base = torch.zeros(2, 10).float()
        base[0, 3] = 1.0
        base[1, 5] = 1.0

        expected = torch.ones(2, 10).float() * smear
        expected[0, 3] = 1.0 - factor
        expected[1, 5] = 1.0 - factor

        self.assertTrue(torch.equal(
            expected, loss._smooth(base)
        ))

    def testSmoothingOneSided(self):
        base = torch.zeros(1, 10).float()
        base[0, 3] = 1.0

        expected = torch.zeros(1, 10).float()
        expected[0, 3] = self.factor

        self.assertTrue(
            torch.equal(self.loss_one_sided._smooth_one_sided(base), expected)
        )

        base = torch.zeros(2, 10).float()
        base[0, 3] = 1.0
        base[1, 5] = 1.0

        expected = torch.zeros(2, 10).float()
        expected[0, 3] = self.factor
        expected[1, 5] = self.factor

        self.assertTrue(torch.equal(
            expected, self.loss._smooth_one_sided(base)
        ))

    def testLoss(self):
        smear = (1-self.factor) / (10 - 1)
        base = torch.zeros(1, 10).float()
        base[0, 3] = 1.0

        expected = torch.ones(1, 10).float() * smear
        expected[0, 3] = self.factor

        self.assertTrue(
            torch.equal(self.loss(base, base), torch.nn.BCELoss()(base, expected))
        )

    def testLossOneSided(self):
        base = torch.zeros(1, 10).float()
        base[0, 3] = 1.0

        expected = torch.zeros(1, 10).float()
        expected[0, 3] = self.factor
        expected_loss = torch.nn.BCELoss()(base, expected)
        res = self.loss_one_sided(base, base)

        self.assertTrue(
            torch.equal(res, expected_loss),
            msg="Loss not equal.\n{}\n{}".format(expected_loss, res)
        )

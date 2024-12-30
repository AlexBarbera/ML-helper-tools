from typing import Callable

import torchvision.io
from torch import Tensor
from torch.utils.data import Dataset
import pandas
import numpy


class TipletDataset(Dataset):
    """
    Creates a `Dataset` from a given index file.
    The index file is a csv with at least 2 columns named \"path\" and \"class\".
    This dataset will randomly select a sample from the same class and from a random class that is different from the
    anchor each time it gets called.
    """

    def __init__(self, path: str, load_fn: Callable[[str], Tensor] = torchvision.io.decode_image):
        """
        Initializes dataset with given index file. Optionally customize load funtion.
        :param path: Path to csv index.
        :param load_fn: Function used to load an image given its path. Should return a tensor.
        """
        super(TipletDataset, self).__init__()

        self.data = pandas.read_csv(path)
        assert "path" not in self.data.columns or "class" not in self.data.columns, \
            "Invalid csv index, expected \"path\" and \"class\" as column names but found: {}".format(self.data.columns)

        self.load_fn = load_fn
        self.total = self.calculate_n()

    def __getitem__(self, item) -> (Tensor, Tensor, Tensor):
        anchor = self.load_fn(self.data["path"][item])
        anchor_class = self.data["class"][item]

        pos_path = self.data[self.data[self.data.index != item]["class"] == anchor_class]["path"].sample(1).item()

        neg_path = self.data[self.data["class"] != anchor_class]["path"].sample(1).item()

        pos = self.load_fn(pos_path)
        neg = self.load_fn(neg_path)

        return anchor, pos, neg

    def calculate_n(self):
        r"""
        Calculates all posible number of samples from given csv index, made from combinations of anchor, positive and
        negative samples chosen at random.
        This is calculated by
        .. math:: f(data) = \sum_i{(len(data_{i}) -1) \times \sum_{j \neq i}{len(data_j)}}
        :return: Theoretical maximum number of data combinations of current dataset.
        """

        groups = numpy.array([len(group) for i, group in self.data.groupby("class")])

        output = 0

        for i in range(len(groups)):
            other = numpy.delete(groups, i).sum()
            inner = groups[i] - 1

            output += inner * other

        return output

    def __len__(self):
        """
        The difference between this and the `self.total` value is that datasets will iterate
        over all the current data once.
        So efectively the REAL size of the dataset to the eyesof the dataloader is the size of the index file.
        :return: The number of instances of the current dataset.
        """
        return len(self.data)

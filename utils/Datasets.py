import itertools
import os.path
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
        assert "path" in self.data.columns and "class" in self.data.columns, \
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


class ABDataset(Dataset):
    """
    Defines a dataset for pairs of images.

    Can take an index file or a root directory. The directory structure can be as follows:
    - data

    -- train

    --- A

    --- B

    -- test

    --- A

    --- B

    -- val

    --- A

    --- B

    And then in the code use it like:
    .. code::

    dataset_train = utils.ABDataset("data/train")
    dataset_test = utils.ABDataset("data/test")
    dataset_test = utils.ABDataset("data/val")

    If using an index file it must be a csv with 2 columns `path` and `class` if you want to generate the pairs
    automatically or `pathA` and `pathB` with you want pre-made pairs.
    """

    def __init__(self, index_path: str, load_fn: Callable[[str], Tensor] = torchvision.io.decode_image,
                 do_matching: bool = False):
        super(ABDataset, self).__init__()

        self.data = None
        if os.path.isfile(index_path) and ".csv" in index_path:
            self.data = pandas.read_csv(index_path)
            if do_matching:
                assert "path" in self.data.columns and "class" in self.data.columns, \
                    "Invalid csv index, expected \"path\" and \"class\" as column names but found: {}".format(
                        self.data.columns)
                assert self.data['class'].unique().size() != 2, (
                    "AB dataset only works with 2 classes, found {}".format(self.data['class'].unique().size())
                )
                self.data = pandas.DataFrame(
                    itertools.product(
                        self.data.loc[self.data['class'] == 0, 'path'].to_list(),
                        self.data.loc[self.data['class'] == 1, 'path'].to_list()
                    ),
                    columns=["pathA", "pathB"]
                )
            else:
                assert "pathA" not in self.data.columns or "pathB" not in self.data.columns, \
                    "Invalid csv index, expected \"pathA\" and \"pathB\" as column names but found: {}".format(
                        self.data.columns
                    )

        else:  # is directory, so iterate and build index
            folders = [x for x in os.listdir(index_path) if os.path.isdir(x)]

            assert "A" in folders and "B" in folders, (
                "Missing folders for `A` and `B` in {}, found {}".format(index_path, folders)
            )

            A = os.listdir(os.path.join(index_path, "A"))
            B = os.listdir(os.path.join(index_path, "B"))

            self.data = pandas.DataFrame(
                itertools.product(A, B),
                columns=["pathA", "pathB"]
            )

        self.load_fn = load_fn

    def __len__(self):
        return self.data.size

    def __getitem__(self, item):
        row = self.data.iloc[item]

        a = row["pathA"]
        b = row["pathB"]

        a = self.load_fn(a)
        b = self.load_fn(b)

        return a, b


class MultiClassMatchingPairDataset(Dataset):
    """
    Dataset that does 1-to-many matching pairs.

    Returns a tuple of (instance, instance, 1 if are_the_same_class else 0)

    Can be done with repetition or without, if `allow_permutated_sample =  True` the dataset will contain "repeated"
    samples but like (A, B, 1) and (B, A, 1) otherwise it will only generate (A, B, 1).

    """

    def __init__(self, path: str, do_pairs: bool = False, include_self: bool = False,
                 allow_permutated_sample: bool = False,
                 load_fn: Callable[[str], Tensor] = torchvision.io.decode_image):
        super(MultiClassMatchingPairDataset, self).__init__()

        self.data = None
        if os.path.isfile(path):
            self.data = pandas.read_csv(path)
            if do_pairs:
                assert "path" in self.data.columns and "class" in self.data.columns, (
                    "Invalid format for index file, expected `path` and `class` as columns, found: {}".format(
                        self.data.columns)
                )
            else:
                assert "pathA" in self.data.columns and "pathB" in self.data.columns, (
                    "Invalid format for index file, expected `pathA` and `pathB` as columns, found: {}".format(
                        self.data.columns)
                )

        else:
            classes = [x for x in os.listdir(path) if os.path.isdir(x)]
            temp = list()

            for c in classes:
                temp.extend([(x, c) for x in os.listdir(os.path.join(path, c)) if os.path.isfile(x)])

            self.data = pandas.DataFrame(temp, columns=["path", "class"])

        if do_pairs:
            self._do_pairs(include_self, allow_permutated_sample)

        self.load_fn = load_fn

    def _do_pairs(self, include_self: bool, allow_duplicates: bool):
        temp = pandas.DataFrame(columns=["pathA", "pathB"])
        classes = self.data["class"].unique()
        it = itertools.combinations_with_replacement(classes, 2) if allow_duplicates else itertools.product(classes,
                                                                                                            classes)

        for c1, c2 in it:
            if c1 == c2 and not include_self:
                continue

            single = pandas.DataFrame(
                itertools.product(
                    [
                        self.data[self.data["class"] == c1]["path"],
                        self.data[self.data["class"] == c2]["path"]
                    ]),
                columns=["pathA", "pathB"]
            )

            single["class"] = int(c1 == c2)

            temp = pandas.concat([temp, single], ignore_index=True)

        self.data = temp

    def __len__(self):
        return self.data.size

    def __getitem__(self, item):
        row = self.data.iloc[item]
        a = row["pathA"]
        b = row["pathB"]
        same = row["class"]

        a = self.load_fn(a)
        b = self.load_fn(b)

        return a, b, same

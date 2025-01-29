from typing import Dict, List, Union, Callable, Tuple, Any, Optional

import numpy
import torch
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class DeepEnergyLoss(torch.nn.Module):
    """
    Maximizes diference between True (typically classified as 1) and False (typically classified as 0) labels.

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
    """

    def __init__(self, alpha: float = 1.0, reduction: bool = True):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred_true, pred_false):
        loss_reg = self.alpha * (pred_true ** 2 + pred_false ** 2).mean()
        loss_div = pred_false.mean() - pred_true.mean()

        if self.reduction:
            return loss_reg + loss_div
        else:
            return loss_reg, loss_div


class BinaryLabelSmoothingLoss(torch.nn.Module):
    """
    Calls a BCELoss after performing label smoothing over a one-hot `expected` vector.
    """

    def __init__(self, target_prob: float, **loss_kw):
        r"""
        Constructs a Label Smoothing BCE loss.
        :param target_prob: Probability of target label after smoothing, rest of labels will have :math:`\frac{1-prob}{n_labels -1}`
        :param loss_kw: Keywords to pass to `torch.nn.BCELoss`
        """
        super().__init__()

        self.factor = target_prob
        self.loss_fn = torch.nn.BCELoss(**loss_kw)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates a copy of tensor `x` and performs smoothing on it.
        :param x: Label tensor to be smooth.
        :return: A copy of smoothed `x`.
        """
        output = (torch.zeros_like(x).fill_((1 - self.factor) / (x.shape[1] - 1))
                  .scatter(1, x.argmax(1, keepdim=True), self.factor))
        return output

    def forward(self, x, y):
        return self.loss_fn(x, self._smooth(y))


class WordTreeTransformation:
    """
    Implementation of WordTree from the YOLO9000 paper, which is a simplification of WordNet.
    WordTree is a Hierarchical representation of classification which allows a more flexible classification tasks.

    YOLO9000 paper: https://arxiv.org/pdf/1612.08242
    """

    def __init__(self, words: List[str], relations: Dict[str, List[str]]):
        """
        Constructor for WordTreeTransform, uses as base `words` and `relations` to transform from a given
        `word` or index to a binary vector.

        While it checks for valid entries (ie. no missing values between `words` and `relations`) we cannot check for
        cyclic relations which will cause it to loop infinitely.

        :param words: A `List` of str representing all the entries in our dataset
        :param relations: A `Dict` matching a single entry of our `words` to a list of other entries of our `words` refering to a parent-children relation
        """

        assert set(words) == (
            set(relations.keys()).union(set(
                [x for value in relations.values() for x in value])
            )), "Some values are missing between words and relations"

        self.total = len(words)
        self.tree = relations
        self.words = words
        self.rwords = {w: i for i, w in enumerate(words)}
        self.rtree = self._gen_reversed_tree()

    def _gen_reversed_tree(self):
        output = dict()
        for key in self.tree.keys():
            for v in self.tree[key]:
                output[v] = key
        for key in set(self.tree.keys()) - set(output.keys()):
            output[key] = key

        return output

    def _get_all_parents(self, x: str) -> List[int]:
        output = list()
        key = x

        if x not in self.rtree:
            raise IndexError("{} not found in index {}".format(x, self.words))

        while key in self.rtree:
            output.append(self._string_to_index(key))

            if key == self.rtree[key]:
                break

            key = self.rtree[key]

        return output

    def _string_to_index(self, x: str) -> int:
        if x not in self.rwords:
            raise IndexError("Index {} not found in {}".format(x, self.words))

        return self.rwords[x]  # self.words.index(x)

    def _index_to_string(self, x: int | torch.Tensor) -> str:
        if isinstance(x, int):
            if x < 0 or x >= self.total:
                raise IndexError("Invalid index {}".format(x))
        elif isinstance(x, torch.Tensor):
            if (x < 0).any() or (x >= self.total).any():
                raise IndexError("Invalid index {}".format(x))

        return self.words[x.int() if isinstance(x, torch.Tensor) else x]

    def logit_to_flattened_tree(self, x: int | str | torch.Tensor | List[str | int]) -> torch.Tensor:
        """
        Create a binary vector for the given input and all of its parents based on the internal dataset.
        :param x: Name or index of one of our dataset tags.
        :return: A binary verctor representing the input tag and all of its parents
        """

        output = None
        if isinstance(x, (int, str)):
            output = torch.zeros(1, self.total).float()

            for index in self._get_all_parents(self._index_to_string(x) if isinstance(x, int) else x):
                output[0, index] = 1.0
        elif isinstance(x, torch.Tensor):
            output = torch.zeros(x.shape[0], self.total).float()
            for i in range(x.shape[0]):

                for index in self._get_all_parents(self._index_to_string(x[i])):
                    output[i, index] = 1.0
        elif isinstance(x, List):
            output = torch.zeros(len(x), self.total).float()
            for i in range(len(x)):

                for index in self._get_all_parents(x[i] if isinstance(x[i], str) else self._index_to_string(x[i])):
                    output[i, index] = 1.0

        return output


class WordTreeLoss(torch.nn.Module):
    def __init__(self, wordtree: WordTreeTransformation, **loss_kw):
        super().__init__()

        self.wt = wordtree
        self.loss = torch.nn.BCELoss(**loss_kw)
        self.levels = [list(x) for x in self.wt.tree.values()]
        self.levels.append([value for key, value in self.wt.rtree.items() if key == value])

        for i in range(len(self.levels)):
            for j in range(len(self.levels[i])):
                self.levels[i][j] = self.wt._string_to_index(self.levels[i][j])

    def _single_loss(self, x, y):
        output = torch.zeros(x.shape[0], 1)
        for i in range(x.shape[0]):
            for level in self.levels:
                output[i] += self.loss(x[i, level], y[i, level])
        return output

    def forward_optimized(self, x, y):
        target = self.wt.logit_to_flattened_tree(y)
        return torch.vmap(func=self._single_loss)(x, target)

    def forward(self, x, y):
        target = self.wt.logit_to_flattened_tree(y)

        output = torch.zeros(x.shape[0], 1)

        for i in range(x.shape[0]):
            for level in self.levels:
                output[i] += self.loss(x[i, level], target[i, level])

        return output

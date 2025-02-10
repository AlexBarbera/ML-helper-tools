import abc
from typing import Tuple

import torch


class AdversarialAttack(abc.ABC):
    """
    Abstract class that wrapps different adversarial attacks.
    """

    @abc.abstractmethod
    def attack(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
        """
        Run the current attack on the given data.
        :param model: Target torch model.
        :param x: Torch tensor input data.
        :param y: Torch tensor target data.
        :param loss: Loss to calculate between output and target.
        :return: A new tensor with the results of the attack.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def find_min_change(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the current attack but returning a single minimum changed tensor.
        :param model: Target torch model.
        :param x: Tensor input data.
        :param y: Tensor target data.
        :param loss: Loss to calculate between output and target.
        :return: A new tensor with the minimum altered tensor and the delta used. If tensor is None means an unsuccessful attack.
        """

        raise NotImplementedError()


class FGSMAttack(AdversarialAttack):
    r"""
    Interface for a FGSM attack (Fast Gradient Sign Method).
    Performs an attack such that :math:`x' =x+\epsilon \times sign(\nabla_x(\theta, x, y)); \text{ } min(\epsilon)`.

    Pytorch tutorial: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    Original Paper: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, min_eps: float = 0.01, max_eps: float = 1.0, delta_steps: int = 20, min_value: int | float = 0,
                 max_value: int | float = 255):
        super().__init__()
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.delta_steps = delta_steps
        self.min_value = min_value
        self.max_value = max_value

    def attack(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
        output = torch.zeros(self.delta_steps, *x.shape)
        pred = model(x)
        l = loss(pred, y)
        l.backward()
        grad = x.grad.data.sign()

        for i, epsilon in enumerate(torch.linspace(self.min_eps, self.max_eps, self.delta_steps)):
            output[i] = torch.clip(x + epsilon * grad, self.min_value, self.max_value)

        return output

    def find_min_change(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        epsilon_final = torch.inf
        output = None

        temp = model(x)
        l = loss(temp, y)
        l.backward()
        grad = x.grad.data.sign()

        max_target = torch.argmax(y)

        for i, epsilon in enumerate(torch.linspace(self.min_eps, self.max_eps, self.delta_steps)):
            res = model(x + epsilon * grad)
            if torch.argmax(res) != max_target:
                epsilon_final = epsilon
                output = x + epsilon * grad
                break

        return output, epsilon_final * grad


class IFGSM(AdversarialAttack):
    r"""
    Performs IFGSM (Iterative-FGSM) (Fast Gradient Sign Method). Also known as Basic Iterative Method (BIL).

    :math:`X_0^adv = X, X^adv_{N+1} = Clip_{X, \epsilon}\{X^adv_N + \alpha \times sign(\nabla_X J(X^adv_N, y_{true})\}`
    """

    def __init__(self, alpha: float = 0.01, max_iters: int = 1000, min_value: int | float = 0,
                 max_value: int | float = 255):
        super().__init__()
        self.alpha = alpha
        self.max_iters = max_iters
        self.min_value = min_value
        self.max_value = max_value

    def _problem_def(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module,
                     y_prime: torch.Tensor | None):
        x_prime = x.clone()
        target = torch.argmax(y)
        n_iters = 1

        if y_prime is None:
            y_prime = torch.tensor(range(y.shape[1]))
            y_prime = y_prime[y_prime != target]
            y_prime = y_prime[torch.randint(0, y.shape[1] - 1, (1,))]
            temp = torch.zeros_like(y)
            temp[y_prime] = 1
            y_prime = temp

        pred = model(x_prime)
        l = loss(pred, y).backward()
        grad = x_prime.grad.data.sign()

        x_prime += self.alpha * grad

        while torch.argmax(model(x_prime)) != target or n_iters > self.max_iters:
            pred = model(x_prime)
            l = loss(pred, y_prime).backward()
            grad = x_prime.grad.data.sign()

            x_prime += self.alpha * grad
            x_prime = x_prime.clip(self.min_value, self.max_value)

            n_iters += 1

        return x_prime, n_iters, n_iters > self.max_iters

    def find_min_change(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        best_x = None
        best_it = torch.inf

        for i in range(y.shape[1]):
            if i == torch.argmax(y):
                continue  # we shouldn't check the expected correct class

            temp = torch.zeros_like(y)
            temp[i] = 1

            xp, iters, solved = self._problem_def(model, x, y, loss, temp)

            if solved and best_it > iters:
                best_it = iters
                best_x = xp

        return best_x, torch.tensor(best_it)

    def attack(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
        return self.find_min_change(model, x, y, loss)[0]


class ILLCM(AdversarialAttack):
    r"""
    Iterative Least-Likely Class Method.
    Iteratively increases likelyhood of the least probable class instead of decreasing max likelyhood.

    :math:`X_0^adv = X, X^adv_{N+1} = Clip_{X, \epsilon}\{X^adv_N - \alpha \times sign(\nabla_X J(X^adv_N, y_{LL})\}`
    """

    def __init__(self, alpha: float, max_iters: int, min_value: int | float = 0, max_value: int | float = 255):
        super().__init__()
        self.alpha = alpha
        self.max_iters = max_iters
        self.min_value = min_value
        self.max_value = max_value

    def _problem_def(self, model, x, y, loss):
        ll = torch.argmin(y)
        yll = torch.zeros_like(y)
        yll[ll] = 1
        iters = 1
        x_adv = x.clone()
        pred = model(x_adv)
        output = list()

        while iters < self.max_iters or torch.argmax(pred) != torch.argmax(y):
            pred = model(x_adv)
            l = loss(pred, yll).backward()
            output.append(x_adv.detach())
            x_adv -= self.alpha * x_adv.grad.sign()

            x_adv = x_adv.clip(self.min_value, self.max_value)

            iters += 1

        return output, iters, iters < self.max_iters

    def attack(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
        return self.find_min_change(model, x, y, loss)[0]

    def find_min_change(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, loss: torch.nn.Module) -> \
            Tuple[torch.Tensor, int, bool]:
        res, it, solved = self._problem_def(model, x, y, loss)
        return res[-1] if solved else None, it, solved

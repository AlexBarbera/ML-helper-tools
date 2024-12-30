import torch


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

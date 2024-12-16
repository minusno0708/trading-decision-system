import torch

class LossWeight():
    def __init__(self, target):
        self.target = target

    def add_weight(self, loss):
        loss= loss.mean(dim=1) * self.weight

        loss = loss.sum() / self.weight.sum()
        
        return loss

    def calc_weight(self, input):
        weight = (input - self.target).abs().mean(dim=1)
        #weight = torch.norm(input - self.target, dim=1)
        weight = 1 / (weight + 1)

        self.weight = weight
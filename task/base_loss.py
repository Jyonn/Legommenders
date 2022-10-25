import torch


class BaseLoss:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss

    def backward(self):
        if self.loss.requires_grad:
            self.loss.backward()

    def get_loss_dict(self) -> dict:
        return self.__dict__


class LossDepot:
    def __init__(self):
        self.table = dict()

    def add(self, loss: BaseLoss):
        loss_dict = loss.get_loss_dict()

        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.table:
                self.table[loss_name] = []
            self.table[loss_name].append(loss_value.detach().cpu().item())

    def summarize(self):
        for loss_name in self.table:
            self.table[loss_name] = torch.tensor(self.table[loss_name]).mean().item()
        return self

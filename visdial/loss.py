import torch
import torch.nn as nn
import torch.nn.functional as F


class FinetuneLoss(nn.Module):
    def __init__(self):
        super(FinetuneLoss, self).__init__()

    def forward(self, scores, batch):
        # scores [BS, NH, NO]
        BS, NH, NO = scores.size()
        relev_round_indices = batch['round_id'] - 1  # Must be -1
        # [BS, 1, NO]
        relev_round_indices = relev_round_indices[:, None, None].repeat(1, 1, NO)
        # [BS, 1, NO]
        scores = torch.gather(scores, 1, relev_round_indices)
        # [BS, NO]
        scores = scores.squeeze(dim=1)
        scores = nn.functional.log_softmax(scores, dim=-1)

        loss = torch.mean((batch['gt_relevance'] * scores)) * (-1)
        return loss


def convert_to_one_hot(target, num_classes):
    """
    :param target: [N, ]
    :param num_classes:
    :return: one_hot [N, num_classes]
    """
    one_hot = torch.zeros(*target.size(), num_classes, device=target.device)
    return one_hot.scatter_(-1, target.unsqueeze(-1), 1.0)


def convert_to_smoothed_label(target, eps=0.1):
    """
    :param target: [N, num_classes] one-hot vector
    :param eps: label smoothing epsilon.
    :return:
    [N, num_classes]
    """
    num_classes = target.size(-1)
    smoothed_label = target * (1 - eps) + (1 - target) * eps / (num_classes - 1)
    return smoothed_label


class VisdialLoss(nn.Module):

    def __init__(self,
                 LS_epsilon=0.1,
                 KD_alpha=None,
                 KD_temperature=None,
                 return_mean=True):
        """
        :param LS_epsilon: epsilon/num_classes will be the soft_target
        :param KD_alpha: 0.5 - 0.5 (balance between KD loss and normal loss)
        :param KD_temperature: 2 - 5 works best.
        :param return_mean:
        """

        super(VisdialLoss, self).__init__()

        self.KD_alpha = KD_alpha
        self.KD_T = KD_temperature
        self.LS_eps = LS_epsilon
        self.return_mean = return_mean

    def forward(self, outputs, target, teacher_outputs=None):
        """
        :param outputs:   [N, num_classes]
        :param target: [N, num_classes]
        :return:
        """
        num_classes = outputs.size(-1)
        batch_size = torch.prod(torch.tensor(outputs.size()[:-1]))

        one_hot_target = convert_to_one_hot(target, num_classes=num_classes)

        if self.LS_eps is not None:
            one_hot_target = convert_to_smoothed_label(one_hot_target, self.LS_eps)

        log_prob = F.log_softmax(outputs, dim=-1)
        loss = -1 * torch.sum(one_hot_target * log_prob)

        if self.KD_T is not None:
            teacher_soft_target = F.softmax(teacher_outputs / self.KD_T, dim=-1)
            log_prob_T = F.log_softmax(outputs / self.KD_T, dim=-1)
            KD_loss = -1 * torch.sum(teacher_soft_target * log_prob_T)
            final_loss = (1 - self.KD_alpha) * loss + self.KD_alpha * (KD_loss * (self.KD_T + self.KD_T))
            if self.return_mean:
                return final_loss / batch_size
            return final_loss

        if self.return_mean:
            return loss / batch_size

        return loss
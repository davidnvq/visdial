import torch
import torch.nn as nn
import torch.nn.functional as F


class FinetuneLoss(nn.Module):
    """
    Compute the loss during the fine-tuning
    """

    def __init__(self):
        super(FinetuneLoss, self).__init__()

    def forward(self, scores, batch):
        """
        Arguments
        ---------
        scores: torch.FloatTensor
            The prediction scores from the model
            Shape [N, num_classes]
        batch: Dictionary
        	The input batch provides the relevance scores
        Returns
        -------
        Loss: torch.FloatTensor
            The computed loss (the mean)
            Shape []
        """
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
    Arguments
    ---------
    target: torch.LongTensor
    	The input tensor of ground truth
    	Shape [N, ]
    Returns
    -------
    one_hot: torch.LongTensor
        The summarized vector of the utility (the context vector for this utility)
        Shape [N, num_classes]
    """
    one_hot = torch.zeros(*target.size(), num_classes, device=target.device)
    return one_hot.scatter_(-1, target.unsqueeze(-1), 1.0)


class DiscLoss(nn.Module):

    def __init__(self, return_mean=True):
        """
        Arguments
        ---------
        return_mean: torch.FloatTensor
            Whether to return the mean
            If not, return the summation of all element loss
        """
        super(DiscLoss, self).__init__()
        self.return_mean = return_mean

    def forward(self, outputs, target):
        """
        Arguments
        ---------
        outputs: torch.FloatTensor
            The prediction scores from the model
            Shape [N, num_classes]
        target: torch.LongTensor
        	The input tensor of ground truth
        	Shape [N, ]
        Returns
        -------
        Loss: torch.FloatTensor
            The computed loss (the summation or the mean)
            Shape []
        """
        num_classes = outputs.size(-1)
        batch_size = torch.prod(torch.tensor(outputs.size()[:-1]))

        one_hot_target = convert_to_one_hot(target, num_classes=num_classes)

        log_prob = F.log_softmax(outputs, dim=-1)
        loss = -1 * torch.sum(one_hot_target * log_prob)

        if self.return_mean:
            return loss / batch_size
        return loss

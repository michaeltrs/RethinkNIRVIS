import torch
import torch.distributed as dist


def accuracy_dist(outputs, labels, class_split, rank, world_size, topk=(1,)):
    """ Computes the precision@k for the specified values of k in parallel
    """
    assert world_size == len(class_split), \
        "world size should equal to the number of class split"
    base = sum(class_split[:rank])
    maxk = max(topk)

    # add each gpu part max index by base
    scores, preds = outputs.topk(maxk, 1, True, True)
    preds += base

    batch_size = labels.size(0)

    # all_gather
    scores_gather = [torch.zeros_like(scores)
                     for _ in range(world_size)]
    dist.all_gather(scores_gather, scores)
    preds_gather = [torch.zeros_like(preds) for _ in range(world_size)]
    dist.all_gather(preds_gather, preds)
    # stack
    _scores = torch.cat(scores_gather, dim=1)
    _preds = torch.cat(preds_gather, dim=1)

    _, idx = _scores.topk(maxk, 1, True, True)
    pred = torch.gather(_preds, dim=1, index=idx)
    pred = pred.t()

    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

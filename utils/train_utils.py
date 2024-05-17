import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 rescale=1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        self.rescale = rescale
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data.div_(self.rescale)
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(alpha=weight_decay, other=p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(other=d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(alpha=momentum, other=buf)
                    else:
                        d_p = buf

                p.data.add_(other=d_p, alpha=-group['lr'])

        return loss


class MarginSoftmaxClassifier(nn.Module):
    def __init__(self, in_features,out_features,s=64.0, m=0.40):
        super(MarginSoftmaxClassifier, self).__init__()
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.kernel)
        self.s = s
        self.m = m

    def forward(self, embedding, label):
        embedding_norm = F.normalize(embedding,dim=1)
        kernel_norm = F.normalize(self.kernel,dim=0)
        cosine = torch.mm(embedding_norm,kernel_norm)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0],
                            cosine.size()[1],
                            device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MarginSoftmaxClassifier(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.40):
        super(MarginSoftmaxClassifier, self).__init__()
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.kernel)
        self.s = s
        self.m = m

    def forward(self, embedding, label):
        embedding_norm = F.normalize(embedding, dim=1)
        kernel_norm = F.normalize(self.kernel, dim=0)
        # print(embedding_norm.shape, kernel_norm.shape)
        # time.sleep(15)
        cosine = torch.mm(embedding_norm, kernel_norm)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0],
                            cosine.size()[1],
                            device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.theta = math.cos(math.pi - m)
        self.sinmm = math.sin(math.pi - m) * m
        self.easy_margin = False
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.kernel)


    def forward(self, embedding, labels):
        embedding_norm = F.normalize(embedding, dim=1)
        kernel_norm = F.normalize(self.kernel, dim=0)
        logits = torch.mm(embedding_norm, kernel_norm)

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class FeatureLoss(nn.Module):
    def __init__(self, weights=None):
        super(FeatureLoss, self).__init__()
        self.criterion = nn.MSELoss()
        if weights is None:
            self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]  # [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        else:
            self.weights = weights

    def forward(self, x, y):
        loss = 0
        for i in range(len(x)):
            loss += self.weights[i] * self.criterion(x[i], y[i])
        return loss



class EKD(nn.Module):
    """ Evaluation-oriented knowledge distillation for deep face recognition, CVPR2022
    """

    def __init__(self):
        super().__init__()
        self.topk = 2000
        self.t = 0.01
        self.anchor = [10, 100, 1000, 10000, 100000, 1000000]
        self.momentum = 0.01
        self.register_buffer('s_anchor', torch.zeros(len(self.anchor)))
        self.register_buffer('t_anchor', torch.zeros(len(self.anchor)))

    def forward(self, g_s, g_t, labels):
        # normalize feature
        class_size = labels.size(0)
        g_s = g_s.view(class_size, -1)
        g_s = F.normalize(g_s)
        classes_eq = (labels.repeat(class_size, 1) == labels.view(-1, 1).repeat(1, class_size))
        # print("classes_eq = ", classes_eq)
        similarity_student = torch.mm(g_s, g_s.transpose(0, 1))
        s_inds = torch.triu(torch.ones(classes_eq.size(), device=g_s.device), 1).bool()

        pos_inds = classes_eq[s_inds]
        # print("pos_inds = ", pos_inds)
        neg_inds = ~classes_eq[s_inds]
        # print("neg_inds = ", neg_inds)
        s = similarity_student[s_inds]
        pos_similarity_student = torch.masked_select(s, pos_inds)
        neg_similarity_student = torch.masked_select(s, neg_inds)
        sorted_s_neg, sorted_s_index = torch.sort(neg_similarity_student, descending=True)

        with torch.no_grad():
            g_t = g_t.view(class_size, -1)
            g_t = F.normalize(g_t)
            similarity_teacher = torch.mm(g_t, g_t.transpose(0, 1))
            t = similarity_teacher[s_inds]
            pos_similarity_teacher = torch.masked_select(t, pos_inds)
            neg_similarity_teacher = torch.masked_select(t, neg_inds)
            sorted_t_neg, _ = torch.sort(neg_similarity_teacher, descending=True)
            length = sorted_s_neg.size(0)
            select_indices = [length // anchor for anchor in self.anchor]
            s_neg_thresholds = sorted_s_neg[select_indices]
            t_neg_thresholds = sorted_t_neg[select_indices]
            self.s_anchor = self.momentum * s_neg_thresholds + (1 - self.momentum) * self.s_anchor
            self.t_anchor = self.momentum * t_neg_thresholds + (1 - self.momentum) * self.t_anchor
        s_pos_kd_loss = self.relative_loss(pos_similarity_student, pos_similarity_teacher)

        s_neg_selected = neg_similarity_student[sorted_s_index[0:self.topk]]
        t_neg_selected = neg_similarity_teacher[sorted_s_index[0:self.topk]]

        s_neg_kd_loss = self.relative_loss(s_neg_selected, t_neg_selected)

        loss = s_pos_kd_loss * 0.02 + s_neg_kd_loss * 0.01

        return loss

    def sigmoid(self, inputs, temp=1.0):
        """ temperature controlled sigmoid
            takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -inputs / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def relative_loss(self, s_similarity, t_similarity):
        s_distance = s_similarity.unsqueeze(1) - self.s_anchor.unsqueeze(0)
        t_distance = t_similarity.unsqueeze(1) - self.t_anchor.unsqueeze(0)

        s_rank = self.sigmoid(s_distance, self.t)
        t_rank = self.sigmoid(t_distance, self.t)

        s_rank_count = s_rank.sum(axis=1, keepdims=True)
        t_rank_count = t_rank.sum(axis=1, keepdims=True)

        s_kd_loss = F.mse_loss(s_rank_count, t_rank_count)
        return s_kd_loss

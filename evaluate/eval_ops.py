import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def evaluate2(gallery_feat, query_feat, labels, fars=[10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2]):

    query_num = query_feat.shape[0]

    similarity = np.dot(query_feat, gallery_feat.T)
    top_inds = np.argsort(-similarity)
    labels = labels.T

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if labels[i, j] == 1:
            correct_num += 1
    top1 = correct_num / query_num
    print("top1 = {:.2%}".format(top1))

    # # calculate top5
    # correct_num = 0
    # for i in range(query_num):
    #     j = top_inds[i, :5]
    #     if any(labels[i, j] == 1.0):
    #         correct_num += 1
    #     # else:
    #     #     print(i,j)
    # top5 = correct_num / query_num
    # print("top5 = {:.4%}".format(top5))

    # # calculate 10
    # correct_num = 0
    # for i in range(query_num):
    #     j = top_inds[i, :10]
    #     if any(labels[i, j] == 1.0):
    #         correct_num += 1
    #     # else:
    #     #     print(i,j)
    # top10 = correct_num / query_num
    # print("top10 = {:.4%}".format(top10))

    labels_ = labels.flatten()
    similarity_ = similarity.flatten()
    fpr, tpr, _ = roc_curve(labels_, similarity_)

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    tpr_fpr_row = []
    for far in fars:
        _, min_index = min(list(zip(abs(fpr - far), range(len(fpr)))))
        tpr_fpr_row.append(tpr[min_index])
        print("TPR {:.2%} @ FAR {:.4%}".format(tpr[min_index], far))

    return [top1], tpr_fpr_row


def evaluate(feat_vis, feat_nir, labels, fars = [10**-5, 10**-4, 10**-3, 10**-2]):

    query_num = feat_nir.shape[0]
    gallery_num = feat_vis.shape[0]

    similarity = np.dot(feat_nir, feat_vis.T)

    top_inds = np.argsort(-similarity)
    labels = labels.T

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if labels[i, j] == 1:
            correct_num += 1
    top1 = correct_num / query_num

    # # calculate top5
    # correct_num = 0
    # for i in range(gallery_num):
    #     j = top_inds[i, 0:5]
    #     if any(labels[i, j] == 1.0):
    #         correct_num += 1
    # top5 = correct_num / query_num
    # # print("top5 = {}".format(top5))
    #
    # # calculate 10
    # correct_num = 0
    # for i in range(gallery_num):
    #     j = top_inds[i, 0:10]
    #     if any(labels[i, j] == 1.0):
    #         correct_num += 1
    # top10 = correct_num / query_num
    # # print("top10 = {}".format(top10))

    labels_ = labels.flatten()
    similarity_ = similarity.flatten()
    fpr, tpr, _ = roc_curve(labels_, similarity_)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    tpr_fpr_row = []
    for fpr_iter in np.arange(len(fars)):
        _, min_index = min(list(zip(abs(fpr - fars[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append(tpr[min_index])

    return [top1], tpr_fpr_row, (fpr, tpr)



def evaluate_nir(feat_vis, feat_nir, labels, fars = [10**-5, 10**-4, 10**-3, 10**-2]):  #, fars):

    query_num = feat_vis.shape[0]
    gallery_num = feat_nir.shape[0]

    similarity = np.dot(feat_vis, feat_nir.T)
    # print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)
    print(top_inds.shape)

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 1]
        if labels[i, j] == 1:
            correct_num += 1
    top1 = correct_num / query_num
    # print("top1 = {}".format(top1))

    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 1:6]
        if any(labels[i, j] == 1.0):
            correct_num += 1
    top5 = correct_num / query_num
    # print("top5 = {}".format(top5))

    # calculate 10
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 1:11]
        if any(labels[i, j] == 1.0):
            correct_num += 1
    top10 = correct_num / query_num
    # print("top10 = {}".format(top10))


    labels_ = labels.flatten()
    similarity_ = similarity.flatten()
    fpr, tpr, _ = roc_curve(labels_, similarity_)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    tpr_fpr_row = []
    for fpr_iter in np.arange(len(fars)):
        _, min_index = min(list(zip(abs(fpr - fars[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append(tpr[min_index])
        # print("TAR %.5f @ FAR %.5f" % (tpr[min_index], fars[fpr_iter]))


    return [top1, top5, top10], tpr_fpr_row, (fpr, tpr)



def plot_roc(FPRTPR, savename, title, names=['rand_init', 'pretrain'], colors=['b', 'r'], styles=['solid', 'dashed'], add_legend=True):
    fig = plt.figure(figsize=(6, 5))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))  # 2 decimal places
    plt.rcParams.update({'font.size': 14})

    ymin = 1.0

    for i, fprtpr in enumerate(FPRTPR):
        fpr, tpr = fprtpr

        ymin_ = tpr[np.argmin(np.abs(fpr - 0.00001))]
        if ymin_ < ymin:
            ymin = ymin_

        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        plt.plot(fpr, tpr, color=colors[i], linestyle=styles[i], lw=1,
                 label='%s' % names[i])

    plt.xlim([10 ** -5, 0.1])
    plt.ylim([ymin, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    # plt.xticks(x_labels)
    plt.yticks(np.linspace(ymin, 1.0, 11, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    if add_legend:
        plt.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(savename)


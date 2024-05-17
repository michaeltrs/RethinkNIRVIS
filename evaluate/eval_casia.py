import numpy as np
import pandas as pd
import os
from evaluate.rgb_ir_face_embed import get_face_embeddings_singlemode
from datetime import datetime
from evaluate import evaluate
import argparse
import re
from glob import glob


def CASIA_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids=[0]):

    def rename_path(s):
        """messy path names, inconsistency between 10-folds and how data are actually saved"""
        s = s.split(".")[0]
        gr, mod, id, img = s.split("\\")
        ext = 'jpg' if (mod == 'VIS') else 'bmp'
        return "%s/%s_%s_%s_%s.%s" % (mod, gr, mod, id, img, ext)

    fars = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    emb = get_face_embeddings_singlemode(arch, checkpoint, root_dir, paths_file, gpu_ids)
    print('face embeddings: ', emb.shape)
    dfeat = emb.shape[1]

    ACC, TARFAR = [], []
    for fold_id in range(1, 11):

        if (fold is not None) and (fold != fold_id):
            continue

        print('fold id: ', fold, fold_id)

        vis = pd.read_csv(os.path.join(basedir, 'protocols', 'vis_gallery_%d.txt' % fold_id), header=None, sep=' ')
        vis_labels = np.array([int(s.split('\\')[-2]) for s in vis[0]])
        vis = vis[0].apply(lambda s: rename_path(s))
        nir = pd.read_csv(os.path.join(basedir, 'protocols', 'nir_probe_%d.txt' % fold_id), header=None, sep=' ')
        nir_labels = np.array([int(s.split('\\')[-2]) for s in nir[0]])
        nir = nir[0].apply(lambda s: rename_path(s))

        labels = np.equal.outer(vis_labels, nir_labels).astype(np.float32)

        feat_vis = np.array([emb.loc[idx].values if idx in emb.index else np.zeros(dfeat) for idx in vis])
        feat_nir = np.array([emb.loc[idx].values if idx in emb.index else np.zeros(dfeat) for idx in nir])

        acc, tarfar, fprtpr = evaluate(feat_vis, feat_nir, labels, fars=fars)
        ACC.append(acc)
        TARFAR.append(tarfar)

    ACC = np.array(ACC)
    acc_mean = ACC.mean(axis=0)
    acc_std = ACC.std(axis=0)

    TARFAR = np.array(TARFAR)
    tar_mean = TARFAR.mean(axis=0)
    tar_std = TARFAR.std(axis=0)

    print('--------------------------------------------------------------------')
    print('CASIA 10-fold validation average performance: ', arch)
    print('- rank-(1, 5, 10) ACC: %s' % acc_mean.tolist())
    print('- TAR@FAR: %s @ %s' % (tar_mean.tolist(), fars))
    print('--------------------------------------------------------------------')

    savename = os.path.join(os.path.dirname(checkpoint), 'CASIA_eval_results.txt')

    with open(savename, 'a') as f:
        f.write('%s, CASIA, %s, %.3f, %.5f, %.5f, %.5f, %.5f, %.5f, %s,  \n'
                % (arch, os.path.basename(checkpoint), acc_mean[0],
                   tar_mean[0], tar_mean[1], tar_mean[2], tar_mean[3], tar_mean[4],
                   datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    return (acc_mean, acc_std), (tar_mean, tar_std)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--architecture', type=str, default=0, help='local_rank')
    parser.add_argument('--checkpoint', type=str, default=0, help='local_rank')
    parser.add_argument('--gpu_ids', type=list, default=[0, 1], action='store',
                        help='gpu ids for running experiment')  # , required=True)
    parser.add_argument('--fold', type=str, default=0, help='evaluation fold')
    parser.add_argument('-l', '--last', action='store_true')  # evaluate only last saved model


    args = parser.parse_args()
    arch = args.architecture
    checkpoint = args.checkpoint
    gpu_ids = [int(i) for i in args.gpu_ids if i.isnumeric()]
    print('gpu ids: ', gpu_ids)
    fold = int(args.fold)
    eval_last = args.last

    basedir = ''
    root_dir = ''
    paths_file = ''

    if fold == 'None':
        fold = None
    elif int(fold) == 0:
        fold = int(re.findall(r"CASIA(\d+)of", checkpoint)[0])
    print('fold: ', fold)

    if os.path.isdir(checkpoint):
        checkpoints = glob(os.path.join(checkpoint, '*backbone.pth'))
        checkpoints.sort(key=os.path.getmtime)
        checkpoints = checkpoints[::-1]

        if eval_last:
            checkpoint = checkpoints[0]
            print('Evaluating checkpoint %s' % checkpoint)
            res = CASIA_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids)
        else:

            for checkpoint in checkpoints:
                print('Evaluating checkpoint %s' % checkpoint)
                res = CASIA_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids)
    else:
        print('Evaluating checkpoint %s' % checkpoint)
        res = CASIA_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids)
        res = CASIA_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids)

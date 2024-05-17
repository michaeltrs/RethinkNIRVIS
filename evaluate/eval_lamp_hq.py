import numpy as np
import pandas as pd
import os
from evaluate.rgb_ir_face_embed import get_face_embeddings_singlemode
from datetime import datetime
from evaluate import evaluate
import argparse
import re
from glob import glob


def LAMPHQ_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids=[0]):

    fars = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    emb = get_face_embeddings_singlemode(architecture=arch,
                                         checkpoint=checkpoint, # warmup=False,
                                         root_dir=root_dir, paths_file=paths_file, gpu_ids=gpu_ids)
    dfeat = emb.shape[1]

    ACC, TARFAR = [], []
    for fold_id in range(1, 11):
        if (fold is not None) and (fold != fold_id):
            continue

        print('fold id: ', fold, fold_id)

        vis = pd.read_csv(os.path.join(basedir, 'Protocol/test', 'gallery_vis%d.txt' % fold_id), header=None, sep=' ')
        nir = pd.read_csv(os.path.join(basedir, 'Protocol/test', 'probe_nir%d.txt' % fold_id), header=None, sep=' ')

        labels = np.equal.outer(vis[1].values, nir[1].values).astype(np.float32)

        feat_vis = np.array([emb.loc[idx].values if idx in emb.index else np.zeros(dfeat) for idx in vis[0]])
        feat_nir = np.array([emb.loc[idx].values if idx in emb.index else np.zeros(dfeat) for idx in nir[0]])

        acc, tarfar, fprtpr = evaluate(feat_vis, feat_nir, labels, fars=fars)  # , fars=[0.1, 0.01, 0.001])
        ACC.append(acc)
        TARFAR.append(tarfar)

    ACC = np.array(ACC)
    acc_mean = ACC.mean(axis=0)
    acc_std = ACC.std(axis=0)

    TARFAR = np.array(TARFAR)
    tar_mean = TARFAR.mean(axis=0)
    tar_std = TARFAR.std(axis=0)

    print('--------------------------------------------------------------------')
    print('LAMP-HQ 10-fold validation average performance: ', arch)
    print('- rank-(1, 5, 10) ACC: %s' % acc_mean.tolist())  # (acc_mean[0], acc_mean[1], acc_mean[2]))
    print('- TAR@FAR: %s @ %s' % (tar_mean.tolist(), fars))
    print('--------------------------------------------------------------------')

    savename = os.path.join(os.path.dirname(checkpoint), 'LAMP-HQ_eval_results_new.txt')

    with open(savename, 'a') as f:
        f.write('%s, LAMP-HQ, %s, %.3f, %.5f, %.5f, %.5f, %.5f, %.5f, %s,  \n'
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

    if fold == 0:
        fold = int(re.findall(r"LAMP-HQ(\d+)of", checkpoint)[0])
    print(fold)

    if os.path.isdir(checkpoint):
        if os.path.exists(os.path.join(os.path.dirname(checkpoint), 'LAMP-HQ_eval_results_new_fold%d.txt' % fold)):
            saved_checkpoints = pd.read_csv(os.path.join(os.path.dirname(checkpoint), 'LAMP-HQ_eval_results_new.txt'), header=None)
            check_if_evaluated = True
        else:
            check_if_evaluated = False

        checkpoints = glob(os.path.join(checkpoint, '*backbone.pth'))
        checkpoints.sort(key=os.path.getmtime)
        checkpoints = checkpoints[::-1]

        if eval_last:
            checkpoint = checkpoints[0]
            print('Evaluating checkpoint %s' % checkpoint)
            res = LAMPHQ_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids)
        else:

            for checkpoint in checkpoints:
                if check_if_evaluated:
                    if checkpoint.split('/')[-1] in saved_checkpoints[2]:
                        continue
                try:
                    print('Evaluating checkpoint %s' % checkpoint)
                    res = LAMPHQ_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids)
                except KeyboardInterrupt:
                    break
                except:
                    continue
    else:
        print('Evaluating checkpoint %s' % checkpoint)
        res = LAMPHQ_eval(checkpoint, root_dir, paths_file, arch, fold, gpu_ids)

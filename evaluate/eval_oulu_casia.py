import numpy as np
import pandas as pd
import os
from evaluate.rgb_ir_face_embed import get_face_embeddings_singlemode
from datetime import datetime
from evaluate import evaluate
import argparse
from glob import glob


def Oulu_CASIA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids=[0]):

    fars = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

    vis_emb = get_face_embeddings_singlemode(arch, checkpoint, root_dir, vis_paths_file, gpu_ids)
    nir_emb = get_face_embeddings_singlemode(arch, checkpoint, root_dir, nir_paths_file, gpu_ids)

    dfeat = vis_emb.shape[1]

    ACC, TARFAR = [], []
    vis = pd.read_csv(vis_paths_file, header=None)  # , sep=' ')
    nir = pd.read_csv(nir_paths_file, header=None)  # , sep=' ')

    labels = np.equal.outer(vis[1].values, nir[1].values).astype(np.float32)

    feat_vis = np.array([vis_emb.loc[idx].values if idx in vis_emb.index else np.zeros(dfeat) for idx in vis[0]])
    feat_nir = np.array([nir_emb.loc[idx].values if idx in nir_emb.index else np.zeros(dfeat) for idx in nir[0]])

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
    print('OULU-CASIA validation performance: ', arch)
    print('- rank-(1, 5, 10) ACC: %s' % acc)  # (acc_mean[0], acc_mean[1], acc_mean[2]))
    print('- TAR@FAR: %s @ %s' % (tarfar, fars))
    print('--------------------------------------------------------------------')

    savename = os.path.join(os.path.dirname(checkpoint), 'OULU-CASIA_eval_results.txt')

    with open(savename, 'a') as f:
        f.write('%s, Oulu-Casia, %s, %.3f, %.5f, %.5f, %.5f, %.5f, %.5f, %s,  \n'
                % (arch, os.path.basename(checkpoint), acc[0],
                   tarfar[0], tarfar[1], tarfar[2], tarfar[3], tarfar[4],
                   datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    return (acc_mean, acc_std), (tar_mean, tar_std)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--architecture', type=str, default=0, help='local_rank')
    parser.add_argument('--checkpoint', type=str, default=0, help='local_rank')
    parser.add_argument('--gpu_ids', type=list, default=[0, 1], action='store',
                        help='gpu ids for running experiment')  # , required=True)
    parser.add_argument('-l', '--last', action='store_true')  # evaluate only last saved model

    args = parser.parse_args()
    arch = args.architecture
    checkpoint = args.checkpoint
    print(args.gpu_ids)
    gpu_ids = [int(i) for i in args.gpu_ids if i.isnumeric()]# if i.isnumeric()]
    eval_last = args.last

    root_dir = ''
    vis_paths_file = ''
    nir_paths_file = ''

    if os.path.isdir(checkpoint):
        checkpoints = glob(os.path.join(checkpoint, '*backbone.pth'))
        checkpoints.sort(key=os.path.getmtime)
        checkpoints = checkpoints[::-1]

        if eval_last:
            checkpoint = checkpoints[0]
            print('Evaluating checkpoint %s' % checkpoint)
            res = Oulu_CASIA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids)
        else:

            for checkpoint in checkpoints:
                print('Evaluating checkpoint %s' % checkpoint)
                res = Oulu_CASIA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids)
    else:
        print('Evaluating checkpoint %s' % checkpoint)
        res = Oulu_CASIA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids)

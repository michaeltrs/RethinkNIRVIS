import numpy as np
import pandas as pd
import os
from evaluate.rgb_ir_face_embed import get_face_embeddings_singlemode
from datetime import datetime
from evaluate import evaluate
import argparse
from glob import glob


def BUUA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids=[0]):

    vis_emb = get_face_embeddings_singlemode(arch, checkpoint, root_dir, vis_paths_file, gpu_ids)
    nir_emb = get_face_embeddings_singlemode(arch, checkpoint, root_dir, nir_paths_file, gpu_ids)

    vis_paths = pd.read_csv(vis_paths_file, header=None)
    vis_paths[1] = vis_paths[0].apply(lambda s: s.split('/')[0])

    nir_paths = pd.read_csv(nir_paths_file, header=None)
    nir_paths[1] = nir_paths[0].apply(lambda s: s.split('/')[0])

    labels = np.equal.outer(vis_paths[1].values, nir_paths[1].values).astype(np.float32)

    fars = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    acc, tarfar, fprtpr = evaluate(vis_emb, nir_emb, labels, fars)

    print('--------------------------------------------------------------------')
    print('BUAA validation performance: ', arch)
    print('- rank-(1, 5, 10) ACC: %s' % acc)  # (acc_mean[0], acc_mean[1], acc_mean[2]))
    print('- TAR@FAR: %s @ %s' % (tarfar, fars))
    print('--------------------------------------------------------------------')


    savename = os.path.join(os.path.dirname(checkpoint), 'BUUA_eval_results.txt')

    with open(savename, 'a') as f:
        f.write('%s, BUAA, %s, %.3f, %.5f, %.5f, %.5f, %.5f, %.5f, %s,  \n'
                % (arch, os.path.basename(checkpoint), acc[0],
                   tarfar[0], tarfar[1], tarfar[2], tarfar[3], tarfar[4],
                   datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    return acc, tarfar


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--architecture', type=str, default=0, help='local_rank')
    parser.add_argument('--checkpoint', type=str, default=0, help='local_rank')
    parser.add_argument('--root_dir', type=str, default=0, help='root directory for the data')
    parser.add_argument('--vis_paths_file', type=str, default=0, help='csv file for vis paths')
    parser.add_argument('--nir_paths_file', type=str, default=0, help='csv file for nir paths')
    parser.add_argument('--gpu_ids', type=list, default=[0, 1], action='store',
                        help='gpu ids for running experiment')
    parser.add_argument('-l', '--last', action='store_true')  # evaluate only last saved model

    args = parser.parse_args()
    arch = args.architecture
    checkpoint = args.checkpoint

    root_dir = ""
    vis_paths_file = ""
    nir_paths_file = ""

    print(args.gpu_ids)
    gpu_ids = [int(i) for i in args.gpu_ids if i.isnumeric()]
    eval_last = args.last

    if os.path.isdir(checkpoint):
        checkpoints = glob(os.path.join(checkpoint, '*backbone.pth'))
        checkpoints.sort(key=os.path.getmtime)
        checkpoints = checkpoints[::-1]

        if eval_last:
            checkpoint = checkpoints[0]
            print('Evaluating checkpoint %s' % checkpoint)
            res = BUUA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids)
        else:
            for checkpoint in checkpoints:
                print('Evaluating checkpoint %s' % checkpoint)
                res = BUUA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids)
    else:
        print('Evaluating checkpoint %s' % checkpoint)
        res = BUUA_eval(checkpoint, root_dir, vis_paths_file, nir_paths_file, arch, gpu_ids)

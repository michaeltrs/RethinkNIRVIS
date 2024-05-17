import os
from glob import glob


def get_paths_fn(dataset):
    """
    Returns a function for obtaining the data paths for each dataset.
    You might need to change the extension of the data paths after basedir, e.g. '**/*.bmp' for buua.
    """
    def get_buua_paths(basedir, relative=False):
        global_paths = glob(os.path.join(basedir, '**/*.bmp'))
        if relative:
            return [os.path.relpath(p, basedir) for p in glob(os.path.join(basedir, '**/*.bmp'))]
        return global_paths

    def get_casia_paths(basedir, relative=False):
        global_paths_jpg = glob(os.path.join(basedir, '*.jpg'))
        global_paths_bmp = glob(os.path.join(basedir, '*.bmp'))

        if relative:
            return [os.path.relpath(p, basedir) for p in glob(os.path.join(basedir, '**/*.jpg'))] + \
                [os.path.relpath(p, basedir) for p in glob(os.path.join(basedir, '**/*.bmp'))]

        return global_paths_jpg + global_paths_bmp

    def get_lamphq_paths(basedir, relative=False):
        global_paths = glob(os.path.join(basedir, '**/*.jpg'))
        if relative:
            return [os.path.relpath(p, basedir) for p in glob(os.path.join(basedir, '**/*.jpg'))]
        return global_paths

    def get_oulucasia_paths(basedir, relative=False):
        global_paths = glob(os.path.join(basedir, '**/**/**/**/*.jpeg'))
        if relative:
            return [os.path.relpath(p, basedir) for p in global_paths]
        return global_paths

    if dataset == 'lamphq':
        return get_lamphq_paths
    elif dataset == 'casia':
        return get_casia_paths
    elif dataset == 'oulucasia':
        return get_oulucasia_paths
    elif dataset == 'buua':
        return get_buua_paths


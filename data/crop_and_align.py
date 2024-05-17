import insightface
import cv2
import numpy as np
from skimage import transform as trans
import os
from data.paths import get_paths_fn
import argparse


def ensure3channel(img):
    if len(img.shape) < 3:
        img = np.concatenate((img, img, img), axis=-1)
    return img


def load_image(impath):
    """
    """
    img = cv2.imread(impath)
    img = ensure3channel(img)
    return img


class FaceCropAlign:
    def __init__(self, target_size=480, crop_size=112, threshold=0.1, gpu_id=0):
        self.target_size = target_size
        self.crop_size = crop_size
        self.threshold = threshold
        self.gpu_id = gpu_id
        self.template = np.array([[0.3488652, 0.2847753],
                                          [0.6613652, 0.2847753],
                                          [0.4995894, 0.4962660],
                                          [0.3784359, 0.7222292],
                                          [0.6215097, 0.7222292]], dtype=np.float32)
        self.template = self.template * self.crop_size

        self.model = insightface.model_zoo.get_model('retinaface_r50_v1')  # ./.insightface
        self.model.prepare(ctx_id=self.gpu_id, nms=0.4)

    def align_to_template(self, img):
        # Detect
        cons_img = np.zeros(shape=(self.target_size, self.target_size, 3)).astype('uint8')
        img_size_max = np.max(img.shape[0:2])
        img_scale = float(self.target_size) / float(img_size_max)

        im = cv2.resize(img, None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
        cons_img[0:im.shape[0], 0:im.shape[1], :] = im

        bbox, landmark = self.model.detect(cons_img, threshold=0.1, scale=1.0)

        if bbox.shape[0] > 1:
            det = bbox[:, 0:4]
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            index = np.argmax(bounding_box_size)
            landmark = landmark[index, :] / img_scale
        elif bbox.shape[0] == 1:
            landmark = landmark[0, :] / img_scale
        else:
            landmark = []

        # Align
        tform = trans.SimilarityTransform()
        tform.estimate(landmark, self.template)
        M = tform.params[0:2, :]
        crop_img = cv2.warpAffine(img, M, (self.crop_size, self.crop_size), borderValue=0.0)

        return crop_img

    def align_and_crop_pairs_from_paths(self, paths, basedir, savedir):

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        dirs = np.unique([p.split('/')[-2] for p in paths])
        for dir in dirs:
            if not os.path.exists(os.path.join(savedir, dir)):
                os.mkdir(os.path.join(savedir, dir))

        saved_data_info_file = os.path.join(savedir, "saved_data_info.csv")

        print("number of image pairs :", len(paths))

        for idx, img_path in enumerate(paths):

            if idx % 1 == 0:
                print("processing file %d of %d" % (idx, len(paths)))

            try:

                global_img_path = os.path.join(basedir, img_path)
                img = load_image(global_img_path)
                print(img.dtype, img.shape)
                img_crop = self.align_to_template(img)

                cv2.imwrite(os.path.join(savedir, img_path), img_crop)

                print(img_crop.shape, os.path.join(savedir, img_path))

                with open(saved_data_info_file, 'a') as fd:
                    fd.write(img_path + ", 1" + "\n")

            except:

                with open(saved_data_info_file, 'a') as fd:
                    fd.write(img_path + ", 0" + "\n")


    def align_and_crop_image(self, impath):
        img = load_image(impath)
        img_crop = self.align_to_template(img)
        return img_crop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face crop and align - dataset preparation')
    parser.add_argument('--dataset', type=str, help='dataset name',
                        choices=['lamphq', 'casia', 'oulucasia', 'buua'])
    parser.add_argument('--savedir', type=str, help='save directory')
    parser.add_argument('--basedir', type=str, help='base directory for the dataset')
    parser.add_argument('--gpu_id', type=int, default=0, help='local device id')

    args = parser.parse_args()
    savedir = args.savedir
    basedir = args.basedir
    dataset = args.dataset
    gpu_id = args.gpu_id

    get_paths = get_paths_fn(dataset)
    paths = get_paths(basedir=basedir, relative=True)

    facecropalign = FaceCropAlign(gpu_id=gpu_id, crop_size=112)
    facecropalign.align_and_crop_pairs_from_paths(paths, basedir, savedir)

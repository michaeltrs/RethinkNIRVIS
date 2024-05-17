from backbones.iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from backbones.lightcnn_v2 import LightCNN_29v2
from backbones.mobilenet import MobileFaceNet


def get_backbone(modelname):
    model_dict = {
        'mobilefacenet': MobileFaceNet,
        'lightcnn29v2': LightCNN_29v2,
        'iresnet18': iresnet18,
        'iresnet34': iresnet34,
        'iresnet50': iresnet50,
        'iresnet100': iresnet100,
                    }
    if modelname in model_dict:
        return model_dict[modelname]
    else:
        print('Model name %s not found in available models' % modelname)

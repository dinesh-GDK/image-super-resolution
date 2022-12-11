import numpy as np
import torch

def RBB2Y(img):
    y = 16.  + (65.481 * img[:, :, 0]  + 128.553 * img[:, :, 1] +  24.966 * img[:, :, 2]) / 255.
    return y

def RGB2YUV(img):

    if len(img.shape) == 4:
        img = img.squeeze(0)

    y  = 16.  + (65.481 * img[:, :, 0]  + 128.553 * img[:, :, 1] +  24.966 * img[:, :, 2]) / 255.
    cb = 128. + (-37.797 * img[:, :, 0] -  74.203 * img[:, :, 1] + 112.000 * img[:, :, 2]) / 255.
    cr = 128. + (112.000 * img[:, :, 0] -  93.786 * img[:, :, 1] -  18.214 * img[:, :, 2]) / 255.
    
    if type(img) == np.ndarray:
        return np.array([y, cb, cr])
    elif type(img) == torch.Tensor:
        return torch.stack((y, cb, cr), 0)
    else:
        raise Exception("Conversion type not supported", type(img))


def YUV2RGB(img):
    r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
    g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
    b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
    return torch.stack([r, g, b], 0).permute(1, 2, 0)

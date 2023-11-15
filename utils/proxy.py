from math import floor, ceil
import torch.nn.functional as F


def reshape_for_Unet_compatibility(layer=4):

    def outer(eval_func):

        def wrapper(*args, **kwargs):

            phi = args[0]['phi']
            mask = args[0]['mask']
            b, _, w, h, d = phi.shape

            padding = [floor((ceil(d / 2 ** layer) * 2 ** layer - d) / 2),
                       ceil((ceil(d / 2 ** layer) * 2 ** layer - d) / 2),
                       floor((ceil(h / 2 ** layer) * 2 ** layer - h) / 2),
                       ceil((ceil(h / 2 ** layer) * 2 ** layer - h) / 2),
                       floor((ceil(w / 2 ** layer) * 2 ** layer - w) / 2),
                       ceil((ceil(w / 2 ** layer) * 2 ** layer - w) / 2)]

            args[0]['phi'] = F.pad(phi, padding)
            args[0]['mask'] = F.pad(mask, padding)

            pred = eval_func(*args, **kwargs)

            if len(pred) == 1:
                b, _, w, h, d = pred.shape
                return pred[:, :, padding[-2]: w - padding[-1], padding[-4]: h - padding[-3],
                       padding[-6]: d - padding[-5]]

            else:
                res = []
                for ele in pred:
                    b, _, w, h, d = ele.shape
                    res.append(ele[:, :, padding[-2]: w - padding[-1], padding[-4]: h - padding[-3],
                               padding[-6]: d - padding[-5]])

                return res

        return wrapper

    return outer


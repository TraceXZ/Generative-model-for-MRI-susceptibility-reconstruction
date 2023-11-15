import torch.nn as nn
import torch
import collections.abc
import numpy as np
from typing import List, Union
import torch.nn.functional as F
from math import ceil, floor
from kornia.filters import spatial_gradient3d


def skew(vector):
    """
    skew-symmetric operator for rotation matrix generation
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def get_rotation_mat(ori1, ori2):
    """
    generating pythonic style rotation matrix
    :param ori1: your current orientation
    :param ori2: orientation to be rotated
    :return: pythonic rotation matrix.
    """
    ori1 = np.array(ori1) if isinstance(ori1, collections.abc.Collection) else ori1
    ori2 = np.array(ori2) if isinstance(ori2, collections.abc.Collection) else ori2
    v = np.cross(ori1, ori2)
    c = np.dot(ori1, ori2)
    mat = np.identity(3) + skew(v) + np.matmul(skew(v), skew(v)) / (1 + c)
    # return torch.from_numpy(mat).float()
    return torch.from_numpy(np.flip(mat).copy()).float().unsqueeze(0).unsqueeze(0)


def get_scaling_mat(scale):
    return torch.diag_embed(torch.tensor(scale)).float()


def cubic_padding(func):
    def wrapper(*args, **kwargs):
        img = args[0]
        b, _, w, h, d = img.shape
        max_dim = max(w, h, d)

        # cubic padding to avoid unreasonably stretching
        padding = [floor((max_dim - d) / 2), ceil((max_dim - d) / 2), floor((max_dim - h) / 2),
                   ceil((max_dim - h) / 2), floor((max_dim - w) / 2), ceil((max_dim - w) / 2)]
        img = F.pad(img, padding)

        return func(*(img, *args[1:]), **kwargs)[:, :, padding[-2]: w + padding[-2], padding[-4]: h + padding[-4],
               padding[-6]: d + padding[-6]]

    return wrapper


@cubic_padding
def affine_transformation(img, affine: Union[List[torch.Tensor], torch.Tensor], pure_translation=None, mode='bilinear'):
    """
    :param img:
    :param affine:
    :param pure_translation:
    :return:
    """
    device = img.device
    b = img.shape[0]

    # add no pure translation
    if pure_translation is None:
        pure_translation = torch.zeros(b, 3, 1).to(device)

    # calculate affine matrices
    affine_mat = torch.eye(3, 3)
    if not isinstance(affine, list):

        affine_mat = affine.squeeze()

    elif len(affine) == 1:

        affine_mat = affine[0]

    else:
        for index in range(len(affine) - 1):
            affine_mat = torch.matmul(affine[index].squeeze().to(device), affine[index + 1].squeeze().to(device))
            affine[index + 1] = affine_mat

    # apply one-step affine transform
    affine_mat = affine_mat.repeat([b, 1, 1]) if len(affine_mat.shape) == 2 else affine_mat
    affine_matrix = torch.cat([affine_mat.to(device), pure_translation], dim=2)
    grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)

    rot_img = F.grid_sample(input=img, grid=grid, mode=mode)
    _, _, w, h, d = rot_img.shape
    return rot_img


def make_coord(shape, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def spatial_gradient(img, rank=0.01):

    grad = spatial_gradient3d(img, order=1)

    def gradient_clip(grad_values):

        top_k_rank = grad_values.topk(round(len(grad_values) * rank))

        value_top_k = top_k_rank[0][-1]
        idx_k = top_k_rank[1]

        grad_values[idx_k] = value_top_k

        return grad_values

    pos_idx = grad > 0
    neg_idx = grad < 0
    clipped_pos_grad = gradient_clip(grad[pos_idx])
    clipped_neg_grad = -gradient_clip(-grad[neg_idx])

    grad[pos_idx] = clipped_pos_grad
    grad[neg_idx] = clipped_neg_grad

    return grad


def spatial_flat(img, rank):

    grad = spatial_gradient3d(img, order=1)
    grad = (grad * grad).sum(dim=2)

    seq = grad[grad != 0].flatten()

    top_k_rank = seq.topk(round(len(seq) * rank))

    k_mini = top_k_rank[0][-1]

    mask1 = torch.zeros_like(img)
    mask2 = torch.zeros_like(img)

    mask1[grad < k_mini] = 1
    mask2[grad > 0] = 1

    return mask1 * mask2 * img

if __name__ == '__main__':

    from handy import *

    img = torch_from_nib_path('H:\\projects\\DIPQSM\\evaluations\\data\\test\\renzo_central_field.nii')
    flat = spatial_flat(img, 0.1)
    save_tensor_as_nii(flat, 'flat')

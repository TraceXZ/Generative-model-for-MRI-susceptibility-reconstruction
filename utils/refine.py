import torch
import numpy
import nibabel as nib
import torch.fft as fft
import torch.nn.functional as F
from myio import *
from qsm_process import truncate_qsm, forward_field_calc, data_fidelity, generate_dipole
from kornia.filters import spatial_gradient3d
import torch.nn as nn
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

kernel_order_2 = nn.Parameter(torch.tensor([[[2, 3, 2],
                  [3, 6, 3],
                  [2, 3, 2]],
                  [[3, 6, 3],
                  [6, -88, 6],
                  [3, 6, 3]],
                  [[2, 3, 2],
                  [3, 6, 3],
                  [2, 3, 2]]
                  ], dtype=torch.float).unsqueeze(0).unsqueeze(0), requires_grad=False) / 26


sobel_order_1 = nn.Parameter(torch.tensor([[
                    [1, 0, 1],
                    [2, 0, -2],
                    [1, 0, -1]],
                    torch.zeros([3, 3]),
                    [[-1, -2, -1],
                     [-2, -4, -2],
                     [-1, -2, -1]]], dtype=torch.float).unsqueeze(0).unsqueeze(0), requires_grad=False)


def dc_refine(pred, phi, dipole, ts=0.05):

    count = 1
    dc = data_fidelity(pred, dipole)
    mask = torch.zeros_like(phi)
    mask[phi != 0] = 1

    while True:

        delta = phi - dc
        pred += truncate_qsm(delta, dipole, ts=0.2)[0]
        dc_next = data_fidelity(pred, dipole)
        if abs(dc_next.norm(1) - dc.norm(1)) / dc.norm(1) < ts:
            break
        dc = dc_next
        count += 1
        print(count)

    return pred * mask


def gd_refine(pred, phi, dipole, step=0.5, ts=1e-2, tv_reg=False, l2_init_reg=False):

    count = 1
    mask = torch.zeros_like(phi)
    mask[phi != 0] = 1
    chi = pred

    while True:

        grad = fft.ifftn(dipole * dipole * fft.fftn(chi)) - fft.ifftn(dipole * fft.fftn(phi))
        grad = grad.real

        # TV reg
        if tv_reg:
            tv_grad = F.conv3d(chi, weight=kernel_order_2.to(chi.device), stride=1, padding=1)
        else:
            tv_grad = torch.zeros_like(grad)

        if grad.norm(1) / chi.norm(1) < ts:
            break

        # L2 initial reg
        if l2_init_reg:
            init_reg = chi - 2 * pred
        else:
            init_reg = torch.zeros_like(grad)

        chi = chi - step * (grad - 0.007 * tv_grad + 0.2 * init_reg)

        count += 1

        print(count)
        if count == 1000:
            break

    return chi * mask


if __name__ == '__main__':

    #
    phi = torch_from_nib_path("G:\data\cosmos0p6\\01EG\\neutral\lfs_resharp_0_smvrad1_cgs_1e-06.nii")[..., 50: -50, 32: -32, 16: -16]
    dipole = generate_dipole(phi.shape, vox=[0.6, 0.6, 0.6])

    mask = torch.zeros_like(phi)
    mask[phi != 0] = 1

    init = phi.clone()
    # res1 = dc_refine(init, phi, dipole)
    res2 = gd_refine(init, phi, dipole, ts=1e-2)
    save_tensor_as_nii(res2, 'res')
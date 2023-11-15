import collections.abc
import math

import torch
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from math import floor, ceil

from deprecated.classic import deprecated
from torch.nn import MSELoss
import nibabel as nib
from typing import List, Union


def reshape_for_Unet_compatibility(layer=4):
    def outer(eval_func):

        def wrapper(*args, **kwargs):

            phi = args[1]
            b, _, w, h, d = phi.shape

            padding = [floor((ceil(d / 2 ** layer) * 2 ** layer - d) / 2),
                       ceil((ceil(d / 2 ** layer) * 2 ** layer - d) / 2),
                       floor((ceil(h / 2 ** layer) * 2 ** layer - h) / 2),
                       ceil((ceil(h / 2 ** layer) * 2 ** layer - h) / 2),
                       floor((ceil(w / 2 ** layer) * 2 ** layer - w) / 2),
                       ceil((ceil(w / 2 ** layer) * 2 ** layer - w) / 2)]

            phi = F.pad(phi, padding)

            pred = eval_func(*(args[0], phi), **kwargs)

            b, _, w, h, d = pred.shape
            return pred[:, :, padding[-2]: w - padding[-1], padding[-4]: h - padding[-3], padding[-6]: d - padding[-5]]

        return wrapper

    return outer


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


def truncate_qsm(phi, dipole, ts=0.2):
    sgn_dipole = torch.sgn(dipole)
    value_dipole = torch.abs(dipole)

    mask = value_dipole > ts
    value_dipole[~mask] = ts
    new_dipole = sgn_dipole * value_dipole
    new_dipole[new_dipole == 0] = ts
    phi_k = fft.fftn(phi)
    tkd_chi = phi_k / new_dipole
    tkd_chi = torch.real(fft.ifftn(tkd_chi))
    mask_phi_k = phi_k * mask
    return tkd_chi, fft.ifftn(mask_phi_k)


def generate_dipole(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1), shift=True, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    vox = np.array(vox) if isinstance(vox, collections.abc.Collection) else vox
    if len(shape) == 5:
        _, _, Nx, Ny, Nz = shape
        FOVx, FOVy, FOVz = vox * shape[2:]
    else:
        Nx, Ny, Nz = shape
        FOVx, FOVy, FOVz = vox * shape
    x = torch.linspace(-Nx / 2, Nx / 2 - 1, steps=Nx)
    y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)
    kx, ky, kz = torch.meshgrid(x/FOVx, y/FOVy, z/FOVz)
    D = 1 / 3 - (kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]) ** 2 / (kx ** 2 + ky ** 2 + kz ** 2)
    D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
    D = D if len(shape) == 3 else D.unsqueeze(0).unsqueeze(0)
    return torch.fft.fftshift(D).to(device) if shift else D.to(device)


# def generate_dipole(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1), shift=True,
#                     device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
#
#     vox = np.array(vox) if isinstance(vox, collections.abc.Collection) else vox
#
#     if len(shape) == 5:
#         _, _, Nx, Ny, Nz = shape
#     else:
#         Nx, Ny, Nz = shape
#
#     FOVx, FOVy, FOVz = vox * (Nx, Ny, Nz)
#     x = torch.linspace(-Nx / 2, Nx / 2 - 1, Nx)
#     y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
#     z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)
#     [kx, ky, kz] = torch.meshgrid(x / FOVx, y / FOVy, z / FOVz)
#     D = 1 / 3 - torch.pow((kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]), 2) / (kx ** 2 + ky ** 2 + kz ** 2)
#     D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
#
#     return torch.fft.fftshift(D).to(device) if shift else D.to(device)


def generate_dipole_img(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1),
                        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):

    # all dimensions should be even

    vox = np.array(vox) if isinstance(vox, collections.abc.Collection) else vox
    if len(shape) == 5:
        _, _, Nx, Ny, Nz = shape
    else:
        Nx, Ny, Nz = shape

    x = torch.linspace(-Nx / 2, Nx / 2 - 1, steps=Nx)
    y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)

    x, y, z = torch.meshgrid(x, y, z)

    x = x * vox[0]
    y = y * vox[1]
    z = z * vox[2]

    d = np.prod(vox) * (3 * (x * z_prjs[0] + y * z_prjs[1] + z * z_prjs[2]) ** 2 - x ** 2 - y ** 2 - z ** 2) \
        / (4 * math.pi * (x ** 2 + y ** 2 + z ** 2) ** 2.5)

    d[torch.isnan(d)] = 0
    d = d if len(shape) == 3 else d.unsqueeze(0).unsqueeze(0)

    return torch.real(fft.fftn(fft.fftshift(d))).to(device)


def forward_field_calc(sus, z_prjs=(0, 0, 1), vox=(1, 1, 1), need_padding=False, tpe='img'):

    device = sus.device
    vox = torch.tensor(vox)
    _, _, Nx, Ny, Nz = sus.size()
    if need_padding:
        sus = F.pad(sus, [Nz//2, Nz//2, Ny//2, Ny//2, Nx//2, Nx//2])
    sz = torch.tensor(sus.size()).to(torch.long)[2:]
    # Nx = sz[0].item()
    # Ny = sz[1].item()
    # Nz = sz[2].item()
    # FOVx, FOVy, FOVz = vox * sz
    # x = torch.linspace(-Nx / 2, Nx / 2 - 1, Nx)
    # y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    # z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)
    # [kx, ky, kz] = torch.meshgrid(x / FOVx, y / FOVy, z / FOVz)
    # D = 1 / 3 - torch.pow((kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]), 2) / (kx ** 2 + ky ** 2 + kz ** 2)
    # D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
    # D = fft.fftshift(D).to(device)
    # ###
    # D = D.unsqueeze(0).unsqueeze(0)
    method = generate_dipole if tpe == 'kspace' else generate_dipole_img
    D = method(sus.shape, z_prjs, vox)
    ###
    field = torch.real(fft.ifftn(D * fft.fftn(sus)))
    return field[:, :, Nx//2: - Nx//2, Ny//2: - Ny//2, Nz//2: - Nz//2] if need_padding else field


# def forward_field_calc(sus, z_prjs=(0, 0, 1), vox=(1, 1, 1), need_padding=False):
#
#     device = sus.device
#     vox = np.array(vox) if isinstance(vox, collections.abc.Collection) else vox
#     shape = sus.shape
#
#     if len(shape) == 5:
#         _, _, Nx, Ny, Nz = shape
#     else:
#         Nx, Ny, Nz = shape
#
#     if need_padding:
#         sus = F.pad(sus, [Nz // 2, Nz // 2, Ny // 2, Ny // 2, Nx // 2, Nx // 2])
#
#     Nx, Ny, Nz = sus.shape[2:]
#     x = torch.linspace(-Nx / 2, Nx / 2 - 1, steps=Nx)
#     y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
#     z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)
#
#     x, y, z = torch.meshgrid(x, y, z)
#
#     x = x * vox[0]
#     y = y * vox[1]
#     z = z * vox[2]
#
#     d = np.prod(vox) * (3 * (x * z_prjs[0] + y * z_prjs[1] + z * z_prjs[2]) ** 2 - x ** 2 - y ** 2 - z ** 2) \
#         / (4 * math.pi * (x ** 2 + y ** 2 + z ** 2) ** 2.5)
#
#     d[torch.isnan(d)] = 0
#     d = d if len(shape) == 3 else d.unsqueeze(0).unsqueeze(0)
#     D = torch.real(fft.fftn(fft.fftshift(d))).to(device)
#     field = torch.real(fft.ifftn(D * fft.fftn(sus)))
#     return field[:, :, Nx // 4: - Nx // 4, Ny // 4: - Ny // 4, Nz // 4: - Nz // 4] if need_padding else field


def model_loss(pred, dipole, field):
    pred_k = fft.fftn(pred, dim=[2, 3, 4])
    field_pk = dipole * pred_k
    field_p = torch.real(fft.ifftn(field_pk, dim=[2, 3, 4]))
    crit = MSELoss(reduction='sum')
    return crit(field_p, field)


@cubic_padding
def rotate(img: torch.Tensor, mat: torch.Tensor):
    """
    rotate a given tensor by specific angles based on rotation matrix.
    No pure translation is applied by default in this implementation.
    This may lead to part of feature being outbound from the FOV, so do padding properly in your own work.
    :param img: tensor to be rotated with shape
    (B, C, Nx, Ny, Nz) in 5-D.
    :param mat: rotation matrix in pythonic format, aka the flipped matlab rotation matrix with shape
    (B, C, 3, 3) in 4-D.
    :return: the rotated tensor.
    """

    b = img.shape[0]
    # add no pure translation
    pure_translation = torch.zeros(b, 3, 1).to(img.device)

    affine_matrix = torch.cat([mat.squeeze(1), pure_translation], dim=2)

    grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)

    rot_img = F.grid_sample(input=img, grid=grid, mode='bilinear')

    return rot_img


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


@deprecated
def reso_manipulate(imgs, reso_list, mode='forward', PATCH_SIZE=(96, 96, 96)):
    res = torch.zeros_like(imgs)
    for index, (img, reso) in enumerate(zip(imgs, reso_list)):
        Nx = round(PATCH_SIZE[0] / reso[0] / 2) * 2
        Ny = round(PATCH_SIZE[1] / reso[1] / 2) * 2
        Nz = round(PATCH_SIZE[2] / reso[2] / 2) * 2

        img = F.interpolate(img.unsqueeze(0), (Nx, Ny, Nz), mode='trilinear')

        delta_x, delta_y, delta_z = PATCH_SIZE[0] - img.shape[2], PATCH_SIZE[1] - img.shape[3], PATCH_SIZE[2] - \
                                    img.shape[4]

        img = F.interpolate(img, PATCH_SIZE, mode='trilinear') \
            if mode == 'forward' else F.pad(img, [delta_z // 2, delta_z // 2, delta_y // 2, delta_y // 2, delta_x // 2,
                                                  delta_x // 2])

        res[index] = img
    return res


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


def torch_from_nib_path(path, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    return torch.from_numpy(nib.load(path).get_fdata()[np.newaxis, np.newaxis]).float().to(device)


def save_tensor_as_nii(tensor, name, vox=(1, 1, 1)):
    return nib.save(nib.Nifti1Image(tensor.squeeze().detach().cpu().numpy(), np.diag((*vox, 1))), name + '.nii')


def save_array_as_nii(arr, name):
    return nib.save(nib.Nifti1Image(arr, np.eye(4)), name + '.nii')


def torch_from_numpy_path(path, device=torch.device('cuda')):
    return torch.from_numpy(np.load(path)[np.newaxis, np.newaxis]).float().to(device)


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


def data_fidelity(chi, Dipole):
    import torch.fft as FFT

    H = chi.size(2)
    L = chi.size(3)
    D = chi.size(4)

    x_k = FFT.fftn(chi, dim=(-3, -2, -1))
    # x_k = x_k.to(device)
    # D = D.to(device)
    x_k = x_k * Dipole  ## forward calculation in k-space.
    x_img = FFT.ifftn(x_k, dim=(-3, -2, -1))

    x_img = torch.real(x_img)

    return x_img


def dipole_convolution_f(sus, dipole):
    b_sus, _, Nx, Ny, Nz = sus.size()
    b_dpl, _, _, _, _ = dipole.size()
    if b_dpl == 1:
        dipole = dipole.repeat([b_sus, 1, 1, 1, 1])
    res = fft.ifftn(dipole * fft.fftn(sus))
    # res = dipole * fft.fftn(sus)
    return res.real

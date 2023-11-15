import torch
import torch.fft as fft
import numpy as np
import collections.abc
from math import floor, pi
import torch.nn.functional as F


def truncate_qsm(phi, dipole=None, vox=None, z_prjs=None, padding=True, ts=0.2):

    """
    calculate TKD QSM based on the given inputs
    :param phi: local field map
    :param dipole: dipole kernel
    :param vox: voxel size
    :param z_prjs: head orientation
    :param padding: padding flag used only when dipole is None
    :param ts: TKD threshold
    :return: TKD-QSM
    """
    if dipole is None:

        if padding:
            px, py, pz = phi.shape[-3] // 2, phi.shape[-2] // 2, phi.shape[-1] // 2
            phi = F.pad(phi,(pz, pz, py, py, px, px))
        else:
            px, py, pz = 0, 0, 0
        dipole = generate_dipole(phi.shape, vox=vox, z_prjs=z_prjs)

    else:
        if vox is not None or z_prjs is not None:
            raise ValueError('vox and z_prjs should be None if dipole is not None')

    sgn_dipole = torch.sgn(dipole)
    value_dipole = torch.abs(dipole)

    mask = value_dipole > ts
    value_dipole[~mask] = ts

    new_dipole = sgn_dipole * value_dipole
    new_dipole[new_dipole == 0] = ts

    phi_k = fft.fftn(phi, dim=[2, 3, 4])

    tkd_chi = phi_k / new_dipole
    tkd_chi = torch.real(fft.ifftn(tkd_chi * mask, dim=[2, 3, 4]))

    mask_phi_k = phi_k * mask

    if padding:
        tkd_chi = tkd_chi[..., px: -px, py: -py, pz: -pz]
        mask_phi_k = mask_phi_k[..., px: -px, py: -py, pz: -pz]

    return tkd_chi, fft.ifftn(mask_phi_k, dim=[2, 3, 4])


def dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]):

    # code from QSMNet repo.

    [Y, X, Z] = np.meshgrid(np.linspace(-np.int64(matrix_size[1] / 2), np.int64(matrix_size[1] / 2) - 1, matrix_size[1]),
                            np.linspace(-np.int64(matrix_size[0] / 2), np.int64(matrix_size[0] / 2) - 1, matrix_size[0]),
                            np.linspace(-np.int64(matrix_size[2] / 2), np.int64(matrix_size[2] / 2) - 1, matrix_size[2]))
    X = X / (matrix_size[0]) * voxel_size[0]
    Y = Y / (matrix_size[1]) * voxel_size[1]
    Z = Z / (matrix_size[2]) * voxel_size[2]
    D = 1 / 3 - np.divide(np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2]),
                          np.square(X) + np.square(Y) + np.square(Z))
    D = np.where(np.isnan(D), 0, D)

    # D = np.roll(D, np.int64(np.floor(matrix_size[0] / 2)), axis=0)
    # D = np.roll(D, np.int64(np.floor(matrix_size[1] / 2)), axis=1)
    # D = np.roll(D, np.int64(np.floor(matrix_size[2] / 2)), axis=2)

    D = np.fft.fftshift(D)
    D = np.float32(D)

    return D


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


def generate_dipole_img(shape, z_prj=(0, 0, 1), vox=(1, 1, 1),
                        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):

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

    d = np.prod(vox) * (3 * (x * z_prj[0] + y * z_prj[1] + z * z_prj[2]) ** 2 - x ** 2 - y ** 2 - z ** 2) \
        / (4 * pi * (x ** 2 + y ** 2 + z ** 2) ** 2.5)

    d[torch.isnan(d)] = 0
    d = d if len(shape) == 3 else d.unsqueeze(0).unsqueeze(0)
    return torch.real(fft.fftn(fft.fftshift(d))).to(device)


def forward_field_calc(sus, z_prjs=(0, 0, 1), vox=(1, 1, 1), need_padding=False):

    device = sus.device

    vox = torch.tensor(vox)

    b, _, Nx, Ny, Nz = sus.size()

    if need_padding:

        sus = F.pad(sus, [Nz // 2, Nz // 2, Ny // 2, Ny // 2, Nx // 2, Nx // 2])

    sz = torch.tensor(sus.size()[-3:]).to(torch.int)

    Nx = sz[0].item()
    Ny = sz[1].item()
    Nz = sz[2].item()

    FOVx, FOVy, FOVz = vox * sz

    x = torch.linspace(-Nx / 2, Nx / 2 - 1, Nx)
    y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)

    [kx, ky, kz] = torch.meshgrid(x / FOVx, y / FOVy, z / FOVz)

    D = 1 / 3 - torch.pow((kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]), 2) / (kx ** 2 + ky ** 2 + kz ** 2)

    D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0

    D = fft.fftshift(D).to(device)

    D = D.unsqueeze(0).repeat([b, 1, 1, 1, 1])

    field = torch.real(fft.ifftn(D * fft.fftn(sus)))

    return field[:, :, Nx // 4: - Nx // 4, Ny // 4: - Ny // 4, Nz // 4: - Nz // 4] if need_padding else field


def data_fidelity(chi, Dipole):

    x_k = fft.fftn(chi, dim=(-3, -2, -1))

    x_k = x_k * Dipole

    x_img = fft.ifftn(x_k, dim=(-3, -2, -1))

    x_img = torch.real(x_img)

    return x_img


def data_consistency(pred, init, dipole, ts):

    pred_k = fft.fftn(pred)
    init_k = fft.fftn(init)

    dc_k = torch.zeros_like(pred_k)
    mask = torch.abs(dipole) < ts
    dc_k[mask] = pred_k[mask]
    dipole[mask] = ts
    dc_k[~mask] = 0.5 * pred_k[~mask] + 0.5 * (init_k / dipole)[~mask]

    return fft.ifftn(dc_k).real


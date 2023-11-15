import math

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torch.fft as FFT
from functools import partial
import torch.nn.functional as F

fft = lambda x: FFT.fftn(x)
ifft = lambda x: FFT.ifftn(x).real


def downsample_by_crop_kspace(img, factor=(1, 1, 1)):

    import torch.fft as fft
    import torch.nn.functional as F

    if all(element == 1 for element in factor):
        return img

    _, _, x, y, z = img.shape
    n_elements = x * y * z

    mask = torch.zeros_like(img)
    mask[img != 0] = 1
    mask = F.interpolate(mask, scale_factor=factor, mode='trilinear')

    x_down, y_down, z_down = int(x * factor[0]), int(y * factor[1]), int(z * factor[2])

    img_freq = fft.fftn(img) / n_elements
    img_shifted_freq = fft.fftshift(img_freq)

    img_crop_shifted_freq = img_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
                            y // 2 - y_down // 2: y // 2 + y_down // 2,
                            z // 2 - z_down // 2: z // 2 + z_down // 2]

    img_crop_freq = fft.ifftshift(img_crop_shifted_freq)

    x_crop = fft.ifftn(img_crop_freq).real * n_elements * (factor[0] * factor[1] * factor[2])

    return x_crop * mask


def reverse_crop(x0, x0_crop, factor, weight=1):

    if all(element == 1 for element in factor):

        return x0_crop

    import torch.fft as fft

    _, _, x, y, z = x0.shape
    _, _, xc, yc, zc = x0_crop.shape

    n_elements = x * y * z
    n_elements_crop = xc * yc * zc

    # mask = torch.zeros_like(x0)
    # mask[x0 != 0] = 1

    x_down, y_down, z_down = int(x * factor[0]), int(y * factor[1]), int(z * factor[2])

    x0_freq = fft.fftn(x0) / n_elements
    x0_crop_freq = fft.fftn(x0_crop) / n_elements_crop

    x0_shifted_freq = fft.fftshift(x0_freq) # [1, 1, 192, 192, 128] 1mm
    x0_crop_shifted_freq = fft.fftshift(x0_crop_freq)

    free_gen = x0_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
               y // 2 - y_down // 2: y // 2 + y_down // 2,
               z // 2 - z_down // 2: z // 2 + z_down // 2]  # [96, 96, 64]

    # normalize
    # mean = free_gen.mean()
    # mean_crop = x0_crop_shifted_freq.mean()
    # x0_crop_shifted_freq = x0_crop_shifted_freq * mean / mean_crop

    x0_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
    y // 2 - y_down // 2: y // 2 + y_down // 2,
    z // 2 - z_down // 2: z // 2 + z_down // 2] = (1 - weight) * free_gen + weight * x0_crop_shifted_freq

    x0_freq = fft.ifftshift(x0_shifted_freq)

    x0 = fft.ifftn(x0_freq).real * n_elements

    return x0


class TVRegularization(nn.Module):
    def __init__(self, weight):
        super(TVRegularization, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, num_channels, height, width, depth = x.size()

        # Calculate horizontal differences
        h_diff = x[:, :, :, :-1, :] - x[:, :, :, 1:, :]

        # Calculate vertical differences
        w_diff = x[:, :, :-1, :, :] - x[:, :, 1:, :, :]

        d_diff = x[:, :, :, :, :-1] - x[:, :, :, :, 1:]

        # Calculate the total variation
        tv = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff)) + torch.sum(torch.abs(d_diff))
        # tv = torch.abs(d_diff)
        # tv = torch.sum(torch.abs(d_diff))

        # Scale the total variation by the weight
        tv_loss = self.weight * tv

        return tv_loss


class Guider(ABC):

    def __init__(self, dim):

        self.phy_model_fn = partial(self.phy_model)
        self.dim = dim
        self.factor = None
        self.guidance = None

    @abstractmethod
    def phy_model(self, *args, **kwargs):

        raise NotImplementedError

    @abstractmethod
    def init_guide(self, x0, xt, xt_prev, **kwargs):

        raise NotImplementedError

    @staticmethod
    def dc_guide(self, x0, xt, xt_prev, **kwargs):

        raise NotImplementedError


    def guidance_init(self, guidance, **kwargs):
        """
        :param guidance: choose from 'init' and 'dc', invalid otherwise
        :param kwargs: include ts, weight, tv, iter_num, step_size
        :return:
        """
        if guidance == 'init':

            self.guidance = partial(self.init_guide, **kwargs)

        elif guidance == 'dc':

            self.guidance = partial(self.dc_guide, **kwargs)

        else:

            raise NotImplementedError

    @staticmethod
    def transdim(func):

        def wrapper(self, x0, **kwargs):

            if self.dim == 2 and len(x0.shape) == 4:
                x0 = x0.permute([1, 2, 3, 0]).unsqueeze(0)
                x0 = func(self, x0, **kwargs)
                x0 = x0.squeeze(1).permute([3, 0, 1, 2])
            else:
                x0 = func(self, x0, **kwargs)
            return x0

        return wrapper


class DipInvGuider(Guider):

    def __init__(self, field, vox=(1, 1, 1), z_prjs=(0, 0, 1), dim=2, is_invivo=True):

        super().__init__(dim)

        field = SuperResDipInvGuider.forward_field_calc(field, vox=vox, z_prjs=z_prjs) if not is_invivo else field

        self.img = field
        self.vox = vox
        self.z_prjs = z_prjs

        self.dipole = DipInvGuider.generate_dipole(field.shape, vox=vox, z_prjs=z_prjs)

        self.phy_model_fn = self.phy_model
        self.tkd_model_fn = self.tkd_model
        self.factor = [1, 1, 1]

    @staticmethod
    def generate_dipole(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1), shift=True,
                        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):

        import numpy as np
        import collections
        from math import floor

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
        kx, ky, kz = torch.meshgrid(x / FOVx, y / FOVy, z / FOVz)
        D = 1 / 3 - (kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]) ** 2 / (kx ** 2 + ky ** 2 + kz ** 2)
        D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
        D = D if len(shape) == 3 else D.unsqueeze(0).unsqueeze(0)
        return torch.fft.fftshift(D).to(device) if shift else D.to(device)

    @staticmethod
    def forward_field_calc(sus, z_prjs=(0, 0, 1), vox=(1, 1, 1), need_padding=False):

        device = sus.device

        sus = sus.squeeze()

        vox = torch.tensor(vox)

        Nx, Ny, Nz = sus.size()

        if need_padding:
            sus = F.pad(sus, [Nz // 2, Nz // 2, Ny // 2, Ny // 2, Nx // 2, Nx // 2])

        mask = torch.zeros_like(sus)
        mask[sus != 0] = 1

        sz = torch.tensor(sus.size()).to(torch.int)

        Nx = sz[0].item()
        Ny = sz[1].item()
        Nz = sz[2].item()

        FOVx, FOVy, FOVz = vox * sz

        x = torch.linspace(-Nx / 2, Nx / 2 - 1, Nx)
        y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
        z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)

        [kx, ky, kz] = torch.meshgrid(x / FOVx, y / FOVy, z / FOVz)

        D = 1 / 3 - torch.pow((kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]), 2) / (kx ** 2 + ky ** 2 + kz ** 2)

        D[math.floor(Nx / 2), math.floor(Ny / 2), math.floor(Nz / 2)] = 0

        D = torch.fft.fftshift(D).to(device)

        D = D.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)

        field = torch.real(ifft(D * fft(sus))) * mask

        return field[:, :, Nx // 4: - Nx // 4, Ny // 4: - Ny // 4, Nz // 4: - Nz // 4] if need_padding else field

    def phy_model(self, x0):

        return ifft(self.dipole * fft(x0)) - self.img

    def tkd_model(self, ts):

        field_k = fft(self.img)

        sgn_dipole = torch.sgn(self.dipole)
        value_dipole = torch.abs(self.dipole)

        mask_k = value_dipole >= ts
        value_dipole[~mask_k] = ts

        new_dipole = sgn_dipole * value_dipole
        new_dipole[new_dipole == 0] = ts

        measurement = field_k / new_dipole
        measurement[~mask_k] = 0

        return ifft(measurement).real, mask_k

    @Guider.transdim
    def init_guide(self, x0, xt=None, xt_prev=None, tv=1e-4, iter_num=5, step_size=0.875, **kwargs):

        tv_reg = TVRegularization(tv)

        x0 = x0.detach().requires_grad_(True)

        for i in range(iter_num):
            grad = torch.autograd.grad(self.phy_model_fn(x0).pow(2).sum() + tv_reg(x0), x0)[0]
            x0 = x0 - step_size * grad

        return x0.real.detach()

    @Guider.transdim
    def dc_guide(self, x0, xt=None, xt_prev=None, ts=0.2, weight=1, tv=0, **kwargs):

        measurement, mask_k = self.tkd_model_fn(ts)

        x0_k = fft(x0)
        m_k = fft(measurement)

        dc = torch.zeros_like(x0_k)
        dc[mask_k == 1] = x0_k[mask_k == 1] * (1 - weight) + m_k[mask_k == 1] * weight
        dc[mask_k == 0] = x0_k[mask_k == 0]

        x0_dc = ifft(dc).real

        tv_reg = TVRegularization(tv)

        x0_dc = x0_dc.detach().requires_grad_(True)

        x0_dc = x0_dc - torch.autograd.grad(tv_reg(x0_dc), x0_dc)[0]

        return x0_dc.real.detach()


class SuperResQSMGuider(Guider):

    def __init__(self, factor, qsm, is_raw_data=True, dim=2):
        super().__init__(dim)

        self.is_raw_data = is_raw_data
        self.factor = factor
        self.img = F.interpolate(qsm, scale_factor=self.factor, mode='trilinear') if self.is_raw_data else qsm

    def phy_model(self, x0):
        return downsample_by_crop_kspace(x0, self.factor) - self.img

    @Guider.transdim
    def init_guide(self, x0, xt=None, xt_prev=None, tv=0, iter_num=5, step_size=0.875, **kwargs):
        # ts, weight, tv, iter_num, step_size

        tv_reg = TVRegularization(tv)

        x0_crop = F.interpolate(x0, scale_factor=self.factor, mode='trilinear')

        x0_crop = x0_crop.detach().requires_grad_(True)

        for i in range(iter_num):
            grad = torch.autograd.grad((x0_crop - self.img).pow(2).sum() + tv_reg(x0_crop), x0_crop)[0]
            x0_crop = x0_crop - step_size * grad

        x0 = reverse_crop(x0, x0_crop.real, self.factor)

        return x0

    @Guider.transdim
    def dc_guide(self, x0, xt=None, xt_prev=None, weight=1, tv=0, **kwargs):

        x0_dc = reverse_crop(x0, self.img, self.factor, weight=weight)
        #x0 [128, 1, 192, 192] -> [1, 1, 192, 192, 128] 1mm, img 2mm [1, 1, 96, 96, 64]
        tv_reg = TVRegularization(tv)

        x0_dc = x0_dc - torch.autograd.grad(tv_reg(x0_dc), x0_dc)[0]

        return x0_dc


class SuperResDipInvGuider(DipInvGuider):

    def __init__(self, factor, field, vox=(1, 1, 1), z_prjs=(0, 0, 1), dim=2, is_raw_data=True, is_upscale=False, is_invivo=True):

        field = field if not is_raw_data else downsample_by_crop_kspace(field, factor)

        field = SuperResDipInvGuider.forward_field_calc(field, vox=vox, z_prjs=z_prjs) if not is_invivo else field

        field = F.interpolate(field, scale_factor=[1 / i for i in factor], mode='trilinear') if is_upscale else field

        super().__init__(field, vox, z_prjs, dim)

        self.factor = factor

        self.is_upscale = is_upscale

    @Guider.transdim
    def dc_guide(self, x0, **kwargs):
        x0_crop = x0 if self.is_upscale else downsample_by_crop_kspace(x0, self.factor)

        x0_crop = super().dc_guide(x0_crop, **kwargs)

        return x0_crop if self.is_upscale else reverse_crop(x0, x0_crop, self.factor)

    @Guider.transdim
    def init_guide(self, x0, xt=None, xt_prev=None, **kwargs):

        x0_crop = x0 if self.is_upscale else downsample_by_crop_kspace(x0, self.factor)

        x0_crop = super().init_guide(x0_crop, **kwargs)

        return x0_crop if self.is_upscale else reverse_crop(x0, x0_crop, self.factor)

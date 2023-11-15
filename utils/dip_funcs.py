import time
import torch.nn as nn
import torch
from utils.handy import truncate_qsm, torch_from_nib_path, save_tensor_as_nii, generate_dipole_img, generate_dipole, forward_field_calc, dipole_convolution_f
import numpy as np
import torch.nn.functional as F
from styleQSM import DIPNet, DIPNetNoSkip
from grad_dip import UnrolledDIPNet, TesterUnrollDIP
from ResNet_yang import ResNet
from unet import Unet
from lpcnn import LPCNN
import torch.fft as fft
from skimage.measure import regionprops
from utils.qsm_process import dipole_kernel
import random
import os


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyLayerNorm3d(nn.LayerNorm):

    def __init__(self, normalized_shape):
        super(MyLayerNorm3d, self).__init__(normalized_shape)

    def forward(self, input):

        input = input.permute([0, 2, 3, 4, 1])
        super(MyLayerNorm3d, self).forward(input)
        return input.permute([0, 4, 1, 2, 3])


class DIPProcess:

    def __init__(self, phi, vox, z_prjs, lr, gamma, step, padding_mode, input_type, is_invivo, device=torch.device('cuda')):

        self.phi = torch_from_nib_path(phi).to(device)

        temp_converted = (self.phi.squeeze().cpu().numpy() * 2 ** 16).astype(np.int16)

        shape = temp_converted.shape
        bbox = np.array(regionprops(temp_converted)[0].bbox)

        frontend = (np.floor(bbox[:3] / 2) * 2).astype(np.int32) - np.array([10, 10, 10])  # make dimensions even and add redundancy of 10
        frontend[frontend < 0] = 0

        backend = np.ceil(bbox[3:] / 2) * 2 + np.array([10, 10, 10])  # make dimensions even and add redundancy of 10
        backend = [min(x, y) for (x, y) in zip(backend.astype(np.int32), shape)]

        bbox = [*frontend.tolist(), *backend]
        self.crop_size = [bbox[2], shape[2] - bbox[5], bbox[1], shape[1] - bbox[4], bbox[0], shape[0] - bbox[3]]

        self.phi = self.phi[:, :, bbox[0]: bbox[3], bbox[1]: bbox[4], bbox[2]: bbox[5]]
        self.mask = torch.zeros_like(self.phi)
        self.mask[self.phi != 0] = 1
        self.dipole = generate_dipole(self.phi.shape, z_prjs, vox).to(device)
        # dipole = dipole_kernel(self.phi.shape, voxel_size=self.vox, B0_dir=self.z_prjs)
        # dipole = torch.from_numpy(dipole).unsqueeze(0).unsqueeze(0).to(self.device)
        if not is_invivo:
            # test only
            self.label = self.phi

            self.phi = forward_field_calc(self.phi, z_prjs=z_prjs, vox=vox, need_padding=True, tpe='kspace')

        self.tkd = truncate_qsm(self.phi, self.dipole, ts=1/8)[0]

        if input_type == 'pure':
            self.input = forward_field_calc(self.tkd, z_prjs=[0, 0, 1], vox=vox, tpe='kspace', need_padding=True) * self.mask
        elif input_type == 'noise':
            self.input = torch.rand_like(self.phi)
        else:
            self.input = self.phi

        self.vox = vox
        self.z_prjs = z_prjs

        seed_torch(3407)
        # torch.cuda.manual_seed(3407)

        # self.model = nn.DataParallel(DIPNetNoSkip(1, 32, 1, use_skip=False, norm=nn.InstanceNorm3d)).to(device)
        self.model = nn.DataParallel(UnrolledDIPNet(1, 32, 1, use_skip=False, norm=nn.InstanceNorm3d)).to(device)
        # self.model = nn.DataParallel(DIPNet(4, 32, 1, encoder_norm=nn.BatchNorm3d, norm=nn.BatchNorm3d, use_skip=False)).to(device)
        # self.model = nn.DataParallel(Unet(4, 16)).to(device)
        # self.model = nn.DataParallel(ResNet(2)).to(device)
        # self.model = nn.DataParallel(TesterUnrollDIP(4, 16, 2, use_skip=True, norm=nn.InstanceNorm3d)).to(device)
        #self.model = nn.DataParallel(LPCNN()).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step, gamma=gamma)

        _, _, xx, yy, zz = self.phi.shape

        if padding_mode == 'none':
            self.padding = 2, 2, 2

        elif padding_mode == 'half':
            self.padding = xx // 2, yy // 2, zz // 2

        elif padding_mode == 'full':
            self.padding = xx, yy, zz

        self.device = device
        self.crit = nn.L1Loss(reduction='sum')

        self.save_path = ''

    def load_state_dict(self, model_state, is_hard=True):

        # self.model.load_state_dict(model_state, is_hard)
        self.model.load_state_dict(model_state, is_hard)

    # advanced funcs, override when using

    def norm(self, pred, dc, epoch):

        return torch.tensor(0).to(self.device)

    def refine(self, epoch, pred):

        return pred

    def criterion(self, pred, phi, dipole, ts=0.2):

        px, py, pz = self.padding

        mask = torch.abs(self.dipole) < ts

        ill_pred = dipole_convolution_f(pred, self.dipole * mask)
        ill_pred = ill_pred * self.mask

        ill_phi = fft.ifftn(fft.fftn(phi) * mask).real * self.mask

        well_pred = fft.ifftn(fft.fftn(pred) * ~mask).real * self.mask
        well_phi = self.tkd * self.mask

        return self.crit(ill_phi, ill_pred) + self.crit(well_pred, well_phi)

    def run(self):

        px, py, pz = self.padding
        _, _, x, y, z = self.phi.shape

        dipole = generate_dipole(((x + 2 * px), (y + 2 * py), (z + 2 * pz)), z_prjs=self.z_prjs, vox=self.vox,
                                 device=self.device).unsqueeze(0).unsqueeze(0)
        # dipole = dipole_kernel(((x + 2 * px), (y + 2 * py), (z + 2 * pz)), voxel_size=self.vox, B0_dir=self.z_prjs)
        # dipole = torch.from_numpy(dipole).unsqueeze(0).unsqueeze(0).to(self.device)
        self.model.eval()

        start_time = time.time()

        for epoch in range(201):

            self.optim.zero_grad()
            # pred_chi = self.model(self.input)[-1] * self.mask
            pred_chi = self.model(self.input, self.phi, self.dipole, epoch)
            # test only
            # add tkd involves way more artifacts than doing iterative GD
            # pred_chi += truncate_qsm(self.phi, self.dipole, ts=0.5)[0]
            pred_chi = pred_chi * self.mask
            #
            dc = dipole_convolution_f(F.pad(pred_chi, [pz, pz, py, py, px, px], mode='circular'), dipole)
            dc = dc[:, :, px: -px, py: -py, pz: -pz] * self.mask

            loss = self.crit(dc, self.phi) + self.norm(pred_chi, dc, epoch)

            # loss = self.criterion(pred_chi, self.phi, dipole)

            loss.backward()
            self.optim.step()
            aft = torch.cuda.memory_allocated()
            self.scheduler.step()

            if epoch % 50 == 0:
                print({'epoch': epoch, 'lr_rate': self.optim.param_groups[0]['lr'], 'loss': loss.item(),
                       'time': int(time.time() - start_time)})
                # print(aft)
                with torch.no_grad():
                    # mean_label = self.label.sum() / (self.label != 0).sum()
                    mean = pred_chi.sum() / (pred_chi != 0).sum()
                    pred_chi = (pred_chi - mean) * self.mask
                    # x0 = F.pad(x0, self.crop_size)
                    pred_chi = F.pad(pred_chi, self.crop_size)
                    # save_tensor_as_nii(x0, self.save_path + '_chi_init_' + str(epoch))
                    save_tensor_as_nii(pred_chi, self.save_path + str(epoch))





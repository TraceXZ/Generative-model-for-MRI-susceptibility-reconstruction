import torch
import torch.nn as nn
from utils.guidance import DipInvGuider, SuperResDipInvGuider, SuperResQSMGuider, downsample_by_crop_kspace
from utils.myio import torch_from_nib_path
from funcs.diffusion import get_beta_schedule, GaussianDiffusion
from models.efficientUnet import SRUnet256 as SRUnet256_3D
from models.efficient_unet2d import SRUnet256 as SRUnet256_2D
from models.unet import UNetModel
from funcs.ddim import get_selection_schedule, DDIM
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def generate(args):

    model_name = args.model
    use_ddim = args.ddim
    state_dict = torch.load(args.state_dict)

    ddpm_steps = args.ddpm_steps
    ddim_step = args.ddim_steps

    data_path = args.data_path
    shape = args.shape
    vox = args.vox
    z_prjs = args.z_prjs
    factor = args.factor

    device = args.device

    task = args.task
    is_raw = args.is_raw  # the data is raw if downsample needed, for SR or SR-DipInv tasks only
    is_invivo = args.is_invivo
    guidance = args.guidance
    ts = args.ts
    weight = args.weight
    tv = args.tv
    iter_num = args.iter_num
    step_size = args.step_size

    # model hyper setting

    if model_name == 'SRUnet256_3D':
        model = SRUnet256_3D()
    elif model_name == 'SRUnet256_2D':
        model = SRUnet256_2D()
    elif model_name == 'Unet-ADM_3D':
        model = UNetModel(image_size=48, in_channels=1, model_channels=64, out_channels=1, channel_mult=(1, 2, 4, 8),
                      num_res_blocks=3, attention_resolutions=[], dropout=0.2,
                      dims=3).cuda()
    model_dim = int(model_name[-2])

    img = torch_from_nib_path(data_path) / 0.15

    mask = torch.zeros_like(img)
    mask[torch.abs(img) > 5e-4] = 1

    if task == 'DipInv':
        guider = DipInvGuider(img, vox=vox, z_prjs=z_prjs, dim=model_dim, is_invivo=is_invivo)

    elif task == 'SRDipInv':
        guider = SuperResDipInvGuider(factor=factor, field=img, vox=vox, z_prjs=z_prjs, dim=model_dim, is_raw_data=is_raw, is_invivo=is_invivo)

    elif task == 'SR':
        guider = SuperResQSMGuider(factor=factor, qsm=img, dim=model_dim, is_raw_data=is_raw)

    else:
        raise ModuleNotFoundError(task + ' is not supported yet')

    guider.guidance_init(guidance, ts=ts, weight=weight, tv=tv, iter_num=iter_num, step_size=step_size, mask_default=mask)

    # adds-on for x_t_next optimization
    guider.s2 = args.step_size2

    device = torch.device(device)

    model = nn.DataParallel(model.to(device))
    model.load_state_dict(state_dict)

    betas = get_beta_schedule('linear', 0.0001, 0.02, ddpm_steps).to(device)

    if use_ddim:

        subsequence = get_selection_schedule('linear', size=ddim_step, timesteps=ddpm_steps)
        diffusion = DDIM(betas, model_mean_type='eps', model_var_type='fixed-small', loss_type='mse', eta=0.5,
                         subsequence=subsequence)

    else:
        diffusion = GaussianDiffusion(betas, model_mean_type='eps', model_var_type='fixed-small', loss_type='mse')

    model.eval()

    pred = diffusion.p_guided_sample(shape, model, device=device, guider=guider) * 0.15
    # pred = diffusion.p_xt_guided_sample(shape, model, device=device, guider=guider, w=weight, mask_default=mask) * 0.15

    save_name = '-'.join([model_name[-2:], task, ''.join(map(str, vox)), ''.join(map(str, z_prjs)),
     'g_' + guidance, 'ts_' + str(ts), 'w_' + str(weight), 'tv_' + str(tv), 'iter_' + str(iter_num),
     'step_' + str(step_size)])

    from utils.myio import save_tensor_as_nii
    save_tensor_as_nii(pred, save_name + '.nii')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='SRUnet256_3D', choices=['SRUnet256_3D', 'SRUnet256_2D', 'Unet-ADM_3D'])
    parser.add_argument('--ddim', type=bool, default=True)
    # parser.add_argument('--state_dict', type=str, default='G:\projects\DiffusionQSm\episodes\e12_norm_testy\chkpt_140.pkl') # for SRUnet256_2D
    parser.add_argument('--state_dict', type=str, default='G:\projects\DiffusionQSm\episodes\episode6\chkpt_130.pkl') # for SRUnet256_3D
    # parser.add_argument('--state_dict', type=str, default='G:\projects\DiffusionQSm\episodes\\network_dataset_tests_ep3\chkpt_270.pkl')

    parser.add_argument('--ddpm_steps', type=int, default=1000)
    parser.add_argument('--ddim_steps', type=int, default=200)

    parser.add_argument('--data_path', type=str, default="G:\projects\DiffusionQSm\evaluations\data\cosmos.nii")
    parser.add_argument('--shape', type=int, nargs=5, default=[1, 1, 192, 192, 64])
    parser.add_argument('--vox', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('--z_prjs', type=float, nargs=3, default=[0, 0, 1])
    parser.add_argument('--factor', type=float, nargs=3, default=[1, 1, 1])

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_name', type=str, default='pred')

    parser.add_argument('--task', type=str, default='SR', choices=['DipInv', 'SRDipInv', 'SR'])
    parser.add_argument('--is_raw', action='store_true', help='the data is raw if downsample needed, for SR or SRDipInv tasks only')
    parser.add_argument('--is_invivo', action='store_true', help='for DipInv or SRDipInv tasks only')

    parser.add_argument('-g', '--guidance', type=str, default='init', choices=['dc', 'init'], help='guidance method choose from [dc, init]')
    parser.add_argument('--ts', type=float, default=0.2, help='threshold for DipInv only')
    parser.add_argument('-w', '--weight', type=float, default=0, help='weight for data consistency')
    parser.add_argument('--tv', type=float, default=1e-4, help='weight for tv loss')
    parser.add_argument('-i', '--iter_num', type=int, default=3, help='number of iterations for iterative optimization')
    parser.add_argument('-s', '--step_size', type=float, default=0.875, help='step_size for iterative optimization')

    args = parser.parse_args()

    import time

    start_time = time.time()

    generate(args)

    print('task finished with elapsed time:' + str((time.time() - start_time) / 60))

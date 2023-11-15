import torch
import torch.nn as nn
from utils.myio import *
import torch.nn.functional as F
import torch.fft as FFT


def model_chi_to_lap_wphase(x):

    b, _, h, w, d = x.shape

    kernel = nn.Parameter(torch.tensor([-1.0, 2.0, -1.0]).reshape(1, 1, 3), requires_grad=False).to(x.device)

    dx = F.conv1d(x.permute(0, 1, 3, 4, 2).reshape(-1, 1, h), kernel, padding=1).reshape([b, 1, w, d, h]).permute(0, 1, 4,
                                                                                                           2, 3)
    dy = F.conv1d(x.permute(0, 1, 2, 4, 3).reshape(-1, 1, w), kernel, padding=1).reshape([b, 1, h, d, w]).permute(0, 1, 2,
                                                                                                           4, 3)
    dz = F.conv1d(x.reshape(-1, 1, d), kernel, padding=1).reshape([b, 1, h, w, d])

    return 1 / 3 * dx + 1 / 3 * dy - 2 / 3 * dz


def laplacian(img):

    import torch
    import torch.nn as nn

    # Define the stencil kernels for each plane
    stencil_kernel1 = torch.tensor([[2, 3, 2], [3, 6, 3], [2, 3, 2]], dtype=torch.float32) / 26
    stencil_kernel2 = torch.tensor([[3, 6, 3], [6, -88, 6], [3, 6, 3]], dtype=torch.float32) / 26
    stencil_kernel3 = torch.tensor([[2, 3, 2], [3, 6, 3], [2, 3, 2]], dtype=torch.float32) / 26

    # Create a PyTorch CNN kernel with 3 input channels and 1 output channel
    cnn_kernel = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False, stride=1)

    # Set the weights of the CNN kernel based on the stencil kernels
    combined_stencil_kernel = torch.stack([stencil_kernel1, stencil_kernel2, stencil_kernel3], dim=0).cuda()
    cnn_kernel.weight.data = torch.unsqueeze(combined_stencil_kernel, dim=0).unsqueeze(0)

    # Set the bias term if needed
    # cnn_kernel.bias.data.fill_(0)  # You can set the bias to zero or adjust it as necessary

    # Test the CNN kernel with some input data (assuming 'input_data' is your input tensor)
    return cnn_kernel(img)


def second_derivative(tensor, direction):

    if direction == 'x':
        # Define the 3D kernel for second derivative w.r.t. x
        kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, -2, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
    elif direction == 'y':
        # Define the 3D kernel for second derivative w.r.t. y
        kernel = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]])
    elif direction == 'z':
        # Define the 3D kernel for second derivative w.r.t. z
        kernel = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[0, 1, 0], [0, -2, 0], [0, 1, 0]]]])
    else:
        raise ValueError("Invalid direction")

    # Expand kernel dimensions to work for any number of channels
    kernel = kernel.expand(tensor.shape[1], -1, -1, -1, -1).float().to(tensor.device)

    # Assuming the tensor is in the shape (batch_size, channels, depth, height, width)
    # Use zero padding to keep dimensions consistent
    return F.conv3d(tensor, kernel, padding=1, groups=tensor.shape[1])


def dip_inv_by_tkd(x_0, field, dipole, ts=0.3, weight=1):
    # dipole = dipole.cpu()
    x_0 = x_0.unsqueeze(-1).transpose(0, -1)
    guidance = field.unsqueeze(-1).transpose(0, -1)

    import torch.fft as FFT

    x_0_freq = FFT.fftn(x_0)

    guidance_freq = FFT.fftn(guidance)
    sgn_dipole = torch.sgn(dipole)

    value_dipole = torch.abs(dipole)

    mask = value_dipole > ts
    value_dipole[~mask] = ts

    new_dipole = sgn_dipole * value_dipole
    new_dipole[new_dipole == 0] = ts

    tkd = guidance_freq / new_dipole * weight + x_0_freq * (1 - weight)
    tkd[~mask] = x_0_freq[~mask]

    x_0 = FFT.ifftn(tkd).cuda().real.transpose(0, -1).squeeze(-1)

    return x_0


from torchvision.models import vgg16


class PerceptualLoss(nn.Module):
    def __init__(self, layers_to_use=None):
        super(PerceptualLoss, self).__init__()

        # Load the pre-trained VGG16 model
        self.vgg = vgg16(weights='DEFAULT').features.eval()

        # By default, use the activations from these layers for the perceptual loss
        if layers_to_use is None:
            layers_to_use = [3, 8, 15, 22, 29]

        self.layers_to_use = layers_to_use

    def forward(self, x, y):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers_to_use:
                loss += torch.nn.functional.mse_loss(x, y)
        return loss


class TVRegularization(nn.Module):

    def __init__(self, weight, dim=(1, 2, 3)):
        super(TVRegularization, self).__init__()

        self.weight = weight

        self.dim = dim

    def forward(self, x):

        tv = 0

        for i in range(1, 4):

            if i == 1:

                tv += torch.sum(torch.abs(x[:, :, :-1, :, :] - x[:, :, 1:, :, :])) if i in self.dim else 0

            elif i == 2:

                tv += torch.sum(torch.abs(x[:, :, :, :-1, :] - x[:, :, :, 1:, :])) if i in self.dim else 0

            elif i == 3:

                tv += torch.sum(torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:])) if i in self.dim else 0

        tv_loss = self.weight * tv

        return tv_loss


# def downsample_by_crop_kspace(x_0, field, dipole, iteration, step_size, ti, factor=(1, 1, 0.5)):
#
#     import torch.fft as fft
#     import torch.nn.functional as F
#
#     _, _, x, y, z = x_0.shape
#     mask = torch.zeros_like(x_0)
#     mask[x_0 != 0] = 1
#     mask = F.interpolate(mask, scale_factor=factor, mode='trilinear')
#
#     x_down, y_down, z_down = int(x * factor[0]), int(y * factor[1]), int(z * factor[2])
#
#     x_0_freq = fft.fftn(x_0)
#     x_0_shifted_freq = fft.fftshift(x_0_freq)
#
#     x_0_crop_shifted_freq = x_0_shifted_freq[:, :, x//2 - x_down//2: x//2 + x_down//2,
#                     y//2 - y_down//2: y//2 + y_down//2,
#                     z//2 - z_down//2: z//2 + z_down//2]
#
#     x_0_crop_freq = fft.ifftshift(x_0_crop_shifted_freq)
#
#     for i in range(iteration):
#
#         grad = torch.autograd.grad((FFT.(dipole * x_0_crop_freq) - field).pow(2).sum() + reg_term(x_0), x_0)[0]
#
#         x_0 = x_0 - step_size * grad
#
#     x_crop = fft.ifftn(x_0_crop_freq).real
#
#     return x_crop * mask


def reverse_crop(x0, x0_crop, vox, weight=1):
    import torch.fft as fft
    import torch.nn.functional as F

    factor = [1 / v for v in vox]

    _, _, x, y, z = x0.shape
    _, _, xc, yc, zc = x0_crop.shape

    n_elements = x * y * z
    n_elements_crop = xc * yc * zc

    mask = torch.zeros_like(x0)
    mask[x0 != 0] = 1

    x_down, y_down, z_down = int(x * factor[0]), int(y * factor[1]), int(z * factor[2])

    x0_freq = fft.fftn(x0) / n_elements
    x0_crop_freq = fft.fftn(x0_crop) / n_elements_crop

    x0_shifted_freq = fft.fftshift(x0_freq)
    x0_crop_shifted_freq = fft.fftshift(x0_crop_freq)

    free_gen = x0_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
               y // 2 - y_down // 2: y // 2 + y_down // 2,
               z // 2 - z_down // 2: z // 2 + z_down // 2]

    x0_shifted_freq[:, :, x // 2 - x_down // 2: x // 2 + x_down // 2,
    y // 2 - y_down // 2: y // 2 + y_down // 2,
    z // 2 - z_down // 2: z // 2 + z_down // 2] = (1 - weight) * free_gen + weight * x0_crop_shifted_freq

    x0_freq = fft.ifftshift(x0_shifted_freq)

    x0 = fft.ifftn(x0_freq).real * n_elements

    return x0 * mask


def downsample_by_crop_kspace(img, factor=(1, 1, 1)):
    import torch.fft as fft
    import torch.nn.functional as F

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


def dip_inv_by_init(x_0, field, dipole, iteration, step_size, ti, generation_dim=2):
    from utils.qsm_process import data_fidelity
    import torch.nn as nn
    reg_term = TVRegularization(0.001)
    # reg_term = nn.Identity()
    *_, h, w, d = field.shape

    if generation_dim == 2:
        x_0 = x_0.permute([1, 2, 3, 0]).unsqueeze(0)
        field = field.permute([1, 2, 3, 0]).unsqueeze(0)
    else:
        bs = x_0.shape[0]
        x_0 = x_0.reshape([3, 4, 2, 1, 48, 48, 48]).permute([0, 4, 1, 5, 2, 6, 3]).reshape([1, 1, h, w, d])

    fft = lambda x: FFT.fftn(x)
    ifft = lambda x: FFT.ifftn(x).real

    # x_0_cropped = downsample_by_crop_kspace(x_0, factor=[1, 0.5, 1])
    # x_0_cropped = F.interpolate(x_0, scale_factor=[0.5, 1, 1])

    # case1
    for i in range(iteration):
        grad = torch.autograd.grad((ifft(dipole * fft(x_0)) - field).pow(2).sum() + reg_term(x_0), x_0)[0]
        # grad = torch.autograd.grad((ifft(dipole * fft(x_0_cropped)) - field).pow(2).sum() + reg_term(x_0), x_0)[0]
        #
        # grad = ifft(dipole * dipole * fft(x_0)) - ifft(dipole * fft(field))

        # if ti < 20:
        # grad += torch.autograd.grad(reg_term(x_0).sum(), x_0)[0]

        x_0 = x_0 - step_size * grad
        x_0 = x_0.real

    # case2

    # x = x_0.detach()
    # x.requires_grad = True
    #
    # for i in range(iteration):
    #
    #     grad = torch.autograd.grad((ifft(dipole * fft(x)) - field).pow(2).sum() + reg_term(x), x)[0]
    #     x = x - step_size * grad
    #     x = x.real
    #
    # x_0 = x_0 * 0.3 + x * 0.7
    #
    if generation_dim == 2:
        x_0 = x_0.squeeze(1).permute([3, 0, 1, 2])
    else:
        x_0 = x_0.reshape([3, 48, 4, 48, 2, 48, 1]).permute([0, 2, 4, 6, 1, 3, 5]).contiguous().reshape(
            [24, 1, 48, 48, 48])
    return x_0


def dip_inv_by_opt(x, x_0, dipole, field, iteration, step_size):
    x_0 = x_0.permute([1, 2, 3, 0]).unsqueeze(0)
    x = x.permute([1, 2, 3, 0]).unsqueeze(0)
    field = field.permute([1, 2, 3, 0]).unsqueeze(0)

    from utils.qsm_process import data_fidelity

    loss_fn = PerceptualLoss().cuda()
    reg = TVRegularization(0.001)

    fft = lambda x: FFT.fftn(x)
    ifft = lambda x: FFT.ifftn(x).real

    for i in range(iteration):
        model_term = (ifft(dipole * fft(x)) - field).pow(2).sum()
        reg_term = reg(x)
        dm_perceptual_term = loss_fn(x.squeeze(1).permute([3, 0, 1, 2]).repeat([1, 3, 1, 1]),
                                     x_0.squeeze(1).permute([3, 0, 1, 2]).repeat([1, 3, 1, 1]))

        grad = torch.autograd.grad(0.7 * model_term + reg_term + dm_perceptual_term, x)[0]

        x = x - step_size * grad

    x_0 = x_0.squeeze(1).permute([3, 0, 1, 2])
    x = x.squeeze(1).permute([3, 0, 1, 2])

    return x


def one_step_by_init(x_0, lap_unwphase):
    from copy import deepcopy
    lap_unwphase = lap_unwphase.permute([1, 2, 3, 0]).unsqueeze(0)
    x_0 = x_0.permute([1, 2, 3, 0]).unsqueeze(0)
    x_0_constant = x_0.clone()
    alpha = 0.1

    for i in range(20):
        # result = (1 / 3) * second_derivative(x_0, 'x') + \
        #          (1 / 3) * second_derivative(x_0, 'y') - \
        #          (2 / 3) * second_derivative(x_0, 'z')
        result = model_chi_to_lap_wphase(x_0)
        grad = alpha * torch.autograd.grad((result - lap_unwphase).pow(2).sum(), x_0)[0]
        x_0 = x_0 - alpha * grad

    x_0 = 0.1 * x_0 + 0.9 * x_0_constant

    return x_0.squeeze(1).permute(3, 0, 1, 2)


def one_step_by_VGG_opt(x, x_0, lap_unwphase):
    loss_fn = PerceptualLoss().cuda()

    grad_perceptual = \
    torch.autograd.grad(loss_fn(x.repeat([1, 3, 1, 1]), x_0.repeat([1, 3, 1, 1])), x, retain_graph=True)[0]
    grad_perceptual = grad_perceptual.unsqueeze(0).permute([0, 2, 3, 4, 1])

    x = x.permute([1, 2, 3, 0]).unsqueeze(0)
    lap_unwphase = lap_unwphase.permute([1, 2, 3, 0]).unsqueeze(0)
    grad_guidance = torch.autograd.grad((model_chi_to_lap_wphase(x) - lap_unwphase).pow(2).sum(), x)[0]

    grad = grad_guidance + grad_perceptual
    x = x - 0.1 * grad
    return x.squeeze(1).permute(3, 0, 1, 2)


def overlapping_grid_indices(x_cond, output_size, r=None):
    _, c, h, w, d = x_cond.shape
    r = 16 if r is None else r
    h_list = []
    w_list = []
    d_list = []
    for i in range(0, h, r):
        if i + output_size >= h:
            h_list.append(h - output_size)
            break
        else:
            h_list.append(i)
    for j in range(0, w, r):
        if j + output_size >= w:
            w_list.append(w - output_size)
            break
        else:
            w_list.append(j)

    for k in range(0, d, r):
        if k + output_size >= d:
            d_list.append(d - output_size)
            break
        else:
            d_list.append(k)

    return h_list, w_list, d_list


if __name__ == '__main__':

    class LoTLayer(nn.Module):
        def __init__(self, conv_x):
            super(LoTLayer, self).__init__()
            self.conv_x = nn.Parameter(conv_x, requires_grad=False)

        def forward(self, phi, mask, TE, B0):
            ## mask: chi mask
            expPhi_r = torch.cos(phi)
            expPhi_i = torch.sin(phi)

            a_r = self.LG(expPhi_r, self.conv_x)  ## first term. (delta(1j * phi)
            a_i = self.LG(expPhi_i, self.conv_x)

            ## b_r = a_r * expPhi_r + a_i * expPhi_i    ## first term  multiply the second term (exp(-1j * phi) = cos(phi) - j * sin(phi)))
            b_i = a_i * expPhi_r - a_r * expPhi_i
            b_i = b_i * mask

            import math
            ## normalization
            b_i = b_i / (B0 * TE * 2 * math.pi)
            b_i = b_i * (3 * 20e-3)

            return b_i

        def LG(self, tensor_image, weight):
            out = F.conv3d(tensor_image, weight, bias=None, stride=1, padding=1)  ## 3 * 3 kernel, padding 1 zeros.

            h, w, d = out.shape[2], out.shape[3], out.shape[4]
            out[:, :, [0, h - 1], :, :] = 0
            out[:, :, :, [0, w - 1], :] = 0
            out[:, :, :, :, [0, d - 1]] = 0
            return out


    qsm = torch_from_nib_path(
        "C:\\Users\\trace\Documents\WeChat Files\s673242975\FileStorage\File\\2023-08\cosmos128.nii")
    qsm = F.interpolate(qsm, scale_factor=[1, 1, 0.5])[:, :, 32: -32, 32: -32, :]

    wp = torch_from_nib_path(
        "C:\\Users\\trace\Documents\WeChat Files\s673242975\FileStorage\File\\2023-08\wph_10ms.nii")
    wp = F.interpolate(wp, scale_factor=[1, 1, 0.5])[:, :, 32: -32, 32: -32, :]
    mask = torch.zeros_like(qsm)
    mask[qsm != 0] = 1

    lap = model_chi_to_lap_wphase(qsm) * mask
    # lap = (lap - lap.min()) / (lap.max() - lap.min()) * mask

    conv_op = [[[1 / 13, 3 / 26, 1 / 13],
                [3 / 26, 3 / 13, 3 / 26],
                [1 / 13, 3 / 26, 1 / 13]],

               [[3 / 26, 3 / 13, 3 / 26],
                [3 / 13, -44 / 13, 3 / 13],
                [3 / 26, 3 / 13, 3 / 26]],

               [[1 / 13, 3 / 26, 1 / 13],
                [3 / 26, 3 / 13, 3 / 26],
                [1 / 13, 3 / 26, 1 / 13]], ]

    conv_op = torch.tensor(conv_op).unsqueeze(0).unsqueeze(0).cuda().float()
    lap_layer = LoTLayer(conv_op)
    lap_unwp = lap_layer(wp, mask, 10, 3) * mask
    lap_unwp = (lap_unwp - lap_unwp.min()) / (lap_unwp.max() - lap_unwp.min()) * mask

    from qsm_process import forward_field_calc

    init = forward_field_calc(qsm) * mask
    init = lap
    init.requires_grad = True

    for i in range(5001):
        # result = (1 / 3) * second_derivative(init, 'x') + \
        #          (1 / 3) * second_derivative(init, 'y') - \
        #          (2 / 3) * second_derivative(init, 'z')
        result = model_chi_to_lap_wphase(init) * mask
        # result = (result - result.min()) / (result.max() - result.min()) * mask

        loss = (result - lap).pow(2).sum()
        grad = 0.1 * torch.autograd.grad(loss, init)[0]
        init = init - grad
        init *= mask
        print(loss)
        if i % 500 == 0:
            save_tensor_as_nii(init, 'init_' + str(i) + '.nii')

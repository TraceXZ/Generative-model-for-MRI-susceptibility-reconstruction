"""
Use the deterministic generative process proposed by Song et al. (2020) [1]
[1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." International Conference on Learning Representations. 2020.
source file: https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py, Ln 342-356
"""  # noqa
import math
from functools import partial

import torch
from .diffusion import GaussianDiffusion, get_beta_schedule
from copy import deepcopy
import torch.nn.functional as F
from utils.guidance import downsample_by_crop_kspace
import torch.fft as fft
from utils.qsm_process import data_fidelity
from torchvision.transforms.functional import crop


__all__ = ["get_selection_schedule", "DDIM"]


# def get_selection_schedule(schedule, size, timesteps):
#     """
#     :param schedule: selection schedule
#     :param size: length of subsequence
#     :param timesteps: total timesteps of pretrained ddpm model
#     :return: a mapping from subsequence index to original one
#     """
#     assert schedule in {"linear", "quadratic"}
#     power = 1 if schedule == "linear" else 2
#     c = timesteps / size ** power
#
#     def subsequence(t: np.ndarray):
#         return np.floor(c * np.power(t + 1, power) - 1).astype(np.int64)
#     return subsequence


def get_selection_schedule(schedule, size, timesteps):
    """
    :param schedule: selection schedule
    :param size: length of subsequence
    :param timesteps: total timesteps of pretrained ddpm model
    :return: subsequence
    """
    assert schedule in {"linear", "quadratic"}

    if schedule == "linear":
        subsequence = torch.arange(0, timesteps, timesteps // size)
    else:
        subsequence = torch.pow(torch.linspace(0, math.sqrt(timesteps * 0.8), size), 2).round().to(torch.int64)  # noqa

    return subsequence


class DDIM(GaussianDiffusion):
    def __init__(self, betas, model_mean_type, model_var_type, loss_type, eta, subsequence):
        super().__init__(betas, model_mean_type, model_var_type, loss_type)
        self.eta = eta  # coefficient between [0, 1] that decides the behavior of generative process
        self.subsequence = subsequence  # subsequence of the accelerated generation

        eta2 = eta ** 2
        try:
            assert not (eta2 != 1. and model_var_type != "fixed-small"), \
                "Cannot use DDIM (eta < 1) with var type other than `fixed-small`"
        except AssertionError:
            # Automatically convert model_var_type to `fixed-small`
            self.model_var_type = "fixed-small"

        self.alphas_bar = self.alphas_bar[subsequence]
        self.alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64).cuda(), self.alphas_bar[:-1]], dim=0)
        self.alphas = self.alphas_bar / self.alphas_bar_prev
        self.betas = 1. - self.alphas
        self.sqrt_alphas_bar_prev = torch.sqrt(self.alphas_bar_prev)

        # q(x_t|x_0)
        # re-parameterization: x_t(x_0, \epsilon_t)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar) * eta2
        self.posterior_logvar_clipped = torch.log(torch.cat([
            self.posterior_var[[1]], self.posterior_var[1:]]).clip(min=1e-20))

        # coefficients to recover x_0 from x_t and \epsilon_t
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1.)

        # coefficients to calculate E[x_{t-1}|x_0, x_t]
        self.posterior_mean_coef2 = torch.sqrt(
            1 - self.alphas_bar - eta2 * self.betas
        ) * torch.sqrt(1 - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_mean_coef1 = self.sqrt_alphas_bar_prev * \
                                    (1. - torch.sqrt(self.alphas) * self.posterior_mean_coef2)

        # for fixed model_var_type's
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large": (
                self.betas, torch.log(torch.cat([self.posterior_var[[1]], self.betas[1:]]).clip(min=1e-20))),
            "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
        }[self.model_var_type]

        self.subsequence = torch.as_tensor(subsequence)

    @torch.inference_mode()
    def p_sample(self, denoise_fn, shape, device=torch.device("cpu"), noise=None, seed=None):
        S = len(self.subsequence)
        B, *_ = shape
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)

        from utils.myio import torch_from_nib_path
        t.fill_(199)
        for ti in range(S - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(_denoise_fn, x_t, t.cuda(), generator=rng)
        # return x_t.reshape([3, 4, 2, 1, 48, 48, 48]).permute([0, 4, 1, 5, 2, 6, 3]).reshape([1, 1, 144, 192, 96])
        return x_t

    def p_sample_3d_guided(self, shape, denoise_fn, guider, device=torch.device("cuda")):

        img_guide = guider.img
        mask = torch.zeros(shape).to(device)
        if guider.factor is not None:
            img_guide = F.interpolate(img_guide, scale_factor=guider.factor)
        mask[img_guide != 0] = 1

        # prepare for diffusion process
        S = len(self.subsequence)
        _, _, h, w, d = shape
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))

        x_t = torch.randn(shape, device=device)

        from utils.condition import overlapping_grid_indices
        h_list, w_list, d_list = overlapping_grid_indices(x_t, output_size=48, r=48)
        corners = [(i, j, k) for i in h_list for j in w_list for k in d_list]

        x_grid_mask = torch.zeros_like(x_t)

        for (hi, wj, dk) in corners:
            x_grid_mask[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += 1

        x_t.requires_grad = True

        mini_bs = 24
        bs = len(corners)
        num_mini_batch = bs // mini_bs + 1

        # iterative reverse process starts
        for ti in range(S - 1, -1, -1):

            eps_t = torch.zeros_like(x_t)
            t = torch.empty((1,), dtype=torch.int64, device=device)
            t.fill_(ti)

            x_t_patch = torch.cat([x_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] for (hi, wj, dk) in corners], dim=0)

            for idx in range(num_mini_batch):

                start = idx * mini_bs
                end = min((idx + 1) * mini_bs, bs)

                with torch.no_grad():
                    out = _denoise_fn(x_t_patch[start: end, :, :, :], t.repeat(end - start))

                for iidx, (hi, wj, dk) in enumerate(corners[start: end]):  # todo: debug here when batch size is 1
                    eps_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += out[iidx, :, :, :, :]

            eps_t = torch.div(eps_t, x_grid_mask)

            t = t[0]

            _clip = (lambda x: x.clamp(-1., 1.))
            x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t))
            #
            x_0 = guider.dc_guide(x_0, weight=1) * mask
            # x_0 = guider.tkd_guide(x_0, 0.2, weight=0.5)

            model_mean, *_ = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)

            if t == 0:
                noise = torch.zeros_like(x_t)
            else:
                noise = torch.empty_like(x_t).normal_()

            model_logvar = self._extract(self.fixed_model_logvar, t, x_t)
            nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

        return x_t.permute([1, 2, 3, 4, 0])

    def p_sample_2d_guidance(self, shape, denoise_fn, guider, device=torch.device("cuda")):

        S = len(self.subsequence)
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))
        img_guide = guider.img

        if guider.factor is not None:
            img_guide = F.interpolate(img_guide, scale_factor=guider.factor)
        mask = torch.zeros(shape).to(device)
        mask[img_guide != 0] = 1

        bs = mask.shape[-1]
        mini_bs = 20

        x_t = torch.randn(shape, device=device).permute([4, 1, 2, 3, 0]).squeeze(-1)
        mask = mask.permute([4, 1, 2, 3, 0]).squeeze(-1)

        x_t.requires_grad = True

        num_mini_batch = bs // mini_bs + 1

        for ti in range(S - 1, -1, -1):

            eps_t = torch.zeros_like(x_t)

            t = torch.empty((1,), dtype=torch.int64, device=device)
            t.fill_(ti)

            for idx in range(num_mini_batch):

                start = idx * mini_bs
                end = min((idx + 1) * mini_bs, bs)

                with torch.no_grad():

                    out = _denoise_fn(x_t[start: end, :, :, :], t.repeat(end - start))

                eps_t[start: end, :, :, :] = out

            t = torch.empty((bs,), dtype=torch.int64, device=device).fill_(ti)

            _clip = (lambda x: x.clamp(-1., 1.))
            x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t))

            #  cond. #2
            from utils.condition import TVRegularization

            tv = TVRegularization(5e-4, dim=[1, 2, 3])

            x_0 = guider.dc_guide(x_0, ts=0.2, weight=1) * mask

            # x_0 = x_0 - 0.875 * torch.autograd.grad(tv(x_0), x_0)[0]

            model_mean, *_ = self.q_posterior_mean_var(x_0, x_t, t)
            model_logvar = self._extract(self.fixed_model_logvar, t, x_t)
            noise = torch.empty_like(x_t).normal_()
            nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

        return x_0 * mask * 0.15

    def p_guided_sample(self, shape, denoise_fn, guider, device=torch.device("cuda")):

        # prepare for diffusion process
        S = len(self.subsequence)
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))

        img_guide = guider.img
        mask = torch.zeros(shape).to(device)

        if guider.__class__.__name__ == "SuperResDipInvGuider":
            if not guider.is_upscale:
                img_guide = F.interpolate(img_guide, scale_factor=[1/i for i in guider.factor])

        elif guider.factor is not None:
            img_guide = F.interpolate(img_guide, scale_factor=[1/i for i in guider.factor])

        assert list(img_guide.shape[2:]) == shape[2:], "shape mismatch between guide and target"

        mask[img_guide != 0] = 1

        x_t = torch.randn(shape, device=device)

        mini_bs = 16

        if guider.dim == 3:

            from condition import overlapping_grid_indices
            h_list, w_list, d_list = overlapping_grid_indices(x_t, output_size=48, r=40)
            corners = [(i, j, k) for i in h_list for j in w_list for k in d_list]

            x_grid_mask = torch.zeros_like(x_t)

            for (hi, wj, dk) in corners:
                x_grid_mask[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += 1

            bs = len(corners)

        elif guider.dim == 2:

            bs = mask.shape[-1]
            x_t = torch.randn(shape, device=device).permute([4, 1, 2, 3, 0]).squeeze(-1)
            mask = mask.permute([4, 1, 2, 3, 0]).squeeze(-1)
            corners = None
            x_grid_mask = torch.ones_like(mask)

        else:
            raise NotImplementedError

        num_mini_batch = math.ceil(bs / mini_bs)

        # iterative reverse process starts
        for ti in range(S - 1, -1, -1):

            eps_t = torch.zeros_like(x_t)

            t = torch.empty((1,), dtype=torch.int64, device=device)
            t.fill_(ti)

            x_t_patch = torch.cat([x_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48]
                                   for (hi, wj, dk) in corners], dim=0) if guider.dim == 3 else x_t

            for idx in range(num_mini_batch):

                start = idx * mini_bs
                end = min((idx + 1) * mini_bs, bs)

                if start == end:
                    break

                with torch.no_grad():
                    out = _denoise_fn(x_t_patch[start: end, :, :, :], t.repeat(end - start))

                if guider.dim == 3:

                    for iidx, (hi, wj, dk) in enumerate(corners[start: end]):  # todo: debug here when batch size is 1
                        eps_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += out[iidx, :, :, :, :]

                else:

                    eps_t[start: end, :, :, :] = out

            eps_t = torch.div(eps_t, x_grid_mask)

            t = t[0] if guider.dim == 3 else torch.empty((bs,), dtype=torch.int64, device=device).fill_(ti)

            _clip = (lambda x: x.clamp(-1., 1.))
            x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t)).detach()

            x_0 = guider.guidance(x_0) * mask

            model_mean, *_ = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)

            if torch.any(t == 0):
                noise = torch.zeros_like(x_t)
            else:
                noise = torch.empty_like(x_t).normal_()

            model_logvar = self._extract(self.fixed_model_logvar, t, x_t)
            nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

        return x_t if guider.dim == 3 else x_t.permute([1, 2, 3, 0]).unsqueeze(0)


    def p_xt_guided_sample(self, shape, denoise_fn, guider, device=torch.device("cuda"), w=0, mask_default=None):

        # prepare for diffusion process
        S = len(self.subsequence)
        subsequence = self.subsequence.to(device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))

        img_guide = guider.img
        mask = torch.zeros(shape).to(device)

        if guider.__class__.__name__ == "SuperResDipInvGuider":
            if not guider.is_upscale:
                img_guide = F.interpolate(img_guide, scale_factor=[1/i for i in guider.factor], mode='trilinear')

        elif guider.factor is not None:
            img_guide = F.interpolate(img_guide, scale_factor=[1/i for i in guider.factor], mode='trilinear')

        assert list(img_guide.shape[2:]) == shape[2:], "shape mismatch between guide and target"

        mask[img_guide != 0] = 1

        x_t = torch.randn(shape, device=device, requires_grad=True)

        mini_bs = 16

        if guider.dim == 3:

            from utils.condition import overlapping_grid_indices
            h_list, w_list, d_list = overlapping_grid_indices(x_t, output_size=48, r=48)
            corners = [(i, j, k) for i in h_list for j in w_list for k in d_list]

            x_grid_mask = torch.zeros_like(x_t)

            for (hi, wj, dk) in corners:
                x_grid_mask[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += 1

            bs = len(corners)

        elif guider.dim == 2:

            bs = mask.shape[-1]
            x_t = torch.randn(shape, device=device, requires_grad=True).permute([4, 1, 2, 3, 0]).squeeze(-1)
            mask = mask.permute([4, 1, 2, 3, 0]).squeeze(-1)
            corners = None
            x_grid_mask = torch.ones_like(mask)

        else:
            raise NotImplementedError

        num_mini_batch = math.ceil(bs / mini_bs)

        # projection
        # forward = partial(F.interpolate, scale_factor=guider.factor, mode='trilinear')
        # backward = partial(F.interpolate, scale_factor=[1/i for i in guider.factor], mode='trilinear')

        forward = partial(downsample_by_crop_kspace, factor=guider.factor)
        from utils.guidance import reverse_crop
        backward = partial(reverse_crop, x0=torch.zeros_like(x_t), factor=guider.factor, weight=1)


        # adds-on keep tune beta_start between [0.01, 0.02]
        weights = get_beta_schedule('linear', 0.0005, 0.003, 200).to(device)
        # iterative reverse process starts
        for ti in range(S - 1, -1, -1):

            eps_t = torch.zeros_like(x_t)

            t = torch.empty((1,), dtype=torch.int64, device=device)
            t.fill_(ti)

            x_t_patch = torch.cat([x_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48]
                                   for (hi, wj, dk) in corners], dim=0) if guider.dim == 3 else x_t

            for idx in range(num_mini_batch):

                start = idx * mini_bs
                end = min((idx + 1) * mini_bs, bs)

                if start == end:
                    break

                with torch.no_grad():
                    out = _denoise_fn(x_t_patch[start: end, :, :, :], t.repeat(end - start))

                if guider.dim == 3:

                    for iidx, (hi, wj, dk) in enumerate(corners[start: end]):  # todo: debug here when batch size is 1
                        eps_t[:, :, hi: hi + 48, wj: wj + 48, dk: dk + 48] += out[iidx, :, :, :, :]

                else:

                    eps_t[start: end, :, :, :] = out

            eps_t = torch.div(eps_t, x_grid_mask)

            t = t[0] if guider.dim == 3 else torch.empty((bs,), dtype=torch.int64, device=device).fill_(ti)

            _clip = (lambda x: x.clamp(-1., 1.))

            x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=eps_t, t=t)) * mask_default

            # for 2D
            # x_0 = x_0.permute([1, 2, 3, 0]).unsqueeze(0)

            # for x0 guidance
            from utils.condition import model_chi_to_lap_wphase, TVRegularization, laplacian
            tv = TVRegularization(0.0005)
            # for i in range(10):
            #
            #     x_0 = x_0 - 0.8 * \
            #           torch.autograd.grad(torch.linalg.norm(model_chi_to_lap_wphase(x_0) * mask_default - guider.img) + tv(x_0), x_0)[0] * mask_default

            # for 2D
            # x_0 = x_0.squeeze(0).permute([3, 0, 1, 2])

            # adds-on
            # x_0_hat = guider.guidance(x_0) * mask
            # model_mean_from_x0_hat, *_ = self.q_posterior_mean_var(x_0=x_0_hat, x_t=x_t, t=t)

            model_mean, *_ = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)

            if torch.any(t == 0):
                noise = torch.zeros_like(x_t)
            else:
                # adds-on
                noise = torch.empty_like(x_t).normal_()

            model_logvar = self._extract(self.fixed_model_logvar, t, x_t)
            nonzero_mask = (t > 0).reshape((-1,) + (1,) * (x_t.ndim - 1)).to(x_t)
            x_t_next = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

            # adds-on
            # x_t_next_hat = model_mean_from_x0_hat + nonzero_mask * torch.exp(0.5 * model_logvar) * noise

            from utils.guidance import reverse_crop
            noisy_measure = self.q_sample(backward(x0_crop=guider.img) * mask_default, t)

            # noisy_measure = self.q_sample(backward(x0_crop=y), t)

            # x_t_next = reverse_crop(x_t_next, noisy_measure, factor=guider.factor, weight=0.01)  # todo: double check the value range

            # DipInv
            # x_t_next = x_t_next - 0.8 * torch.autograd.grad(torch.linalg.norm(torch.fft.ifftn(guider.dipole * torch.fft.fftn(x_0)) - guider.img), x_t)[0]

            # SR
            # x_t_next = x_t_next - guider.s2 * torch.autograd.grad(torch.linalg.norm(forward(x_0) - guider.img), x_t)[0] * mask_default

            # SRDipInv
            # x_t_next = x_t_next - 1 * torch.autograd.grad(torch.linalg.norm(torch.fft.ifftn(guider.dipole * torch.fft.fftn(forward(x_0))) - guider.img), x_t)[0]

            # Single-Step QSM test
            from utils.condition import model_chi_to_lap_wphase
            x_t_next = x_t_next - 1 * torch.autograd.grad(torch.linalg.norm(model_chi_to_lap_wphase(x_0) - guider.img), x_t)[0] * mask_default

            from utils.qsm_process import forward_field_calc
            # x_t_next = x_t_next - guider.s2 * torch.autograd.grad(torch.linalg.norm(forward_field_calc(forward(x_0), guider.z_prjs, guider.vox, need_padding=False) - guider.img), x_t)[0]
            # x_t projection
            # yt_hat_mean = torch.mean(x_t_next)
            # yt_mean = torch.mean(noisy_measure)
            # yt = noisy_measure - yt_mean + yt_hat_mean
            # x_t_next = (1-weights[t]) * x_t_next + weights[t] * yt
            # x_t_next = (1-w) * x_t_next + w * noisy_measure  # the larger weight, the more noisy, which means yt_hat is less noisy than yt (trilinear case)
            # if t <= 10:
            #     from utils.myio import save_tensor_as_nii
            #     save_tensor_as_nii(x_t, 'xt_' + str(t.item()) + '.nii')
            #     save_tensor_as_nii(noisy_measure, 'noisy_measure_' + str(t.item()) + '.nii')
            # x_t_next, _ = condition.conditioning(x_t, x_t_next, x_0, guider.img)

            x_t = x_t_next

        return x_0 * mask_default if guider.dim == 3 else x_t.permute([1, 2, 3, 0]).unsqueeze(0)

    @staticmethod
    def from_ddpm(diffusion, eta, subsequence):
        return DDIM(**{
            k: diffusion.__dict__.get(k, None)
            for k in ["betas", "model_mean_type", "model_var_type", "loss_type"]
        }, eta=eta, subsequence=subsequence)


if __name__ == "__main__":

    subsequence = get_selection_schedule("linear", 10, 1000)
    print(subsequence)
    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000)
    diffusion = GaussianDiffusion(betas, "eps", "fixed-small", "mse")
    print(diffusion.__dict__)
    print(DDIM.from_ddpm(diffusion, eta=0., subsequence=subsequence).__dict__)

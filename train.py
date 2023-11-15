import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data.distributed import DistributedSampler
from funcs.diffusion import GaussianDiffusion, get_beta_schedule
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v/self.count))
        return dict(avg_stats)


class Trainer:

    def __init__(self, model,
                 optimizer,
                 diffusion: GaussianDiffusion,
                 epochs,
                 trainloader,
                 sampler=None,
                 scheduler=None,
                 num_accum=1,
                 use_ema=False,
                 grad_norm=1.0,
                 shape=None,
                 device=torch.device('cuda'),
                 chkpt_intv=10,
                 image_intv=1,
                 num_save_images=8,
                 distributed=False,
                 rank=0,
                 dry_run=False):

        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.epochs = epochs
        self.start_epoch = 1
        self.trainloader = trainloader
        self.sampler = sampler

        if shape is None:

            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = scheduler
        self.num_accum = num_accum  # an argument for mini-batch
        self.grad_norm = grad_norm
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.image_intv = image_intv
        self.num_save_images = num_save_images

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.rank = rank
        self.dry_run = dry_run
        self.is_leader = rank == 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.sample_seed = torch.initial_seed() + self.rank  # device-specific seed

        self.use_ema = use_ema
        self.stats = RunningStatistics(loss=None)
        self.writer = SummaryWriter('runs')

    @property
    def timesteps(self):
        return self.diffusion.timesteps

    @property
    def current_stats(self):
        return self.stats.extract()

    def loss(self, x):

        t = torch.randint(self.timesteps, size=(x.shape[0], ), dtype=torch.int64, device=self.device)
        return self.diffusion.train_losses(self.model, x, t)

    def step(self, x, global_steps=1):

        loss = self.loss(x).mean()
        loss.div(self.num_accum).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.detach()
        self.stats.update(x.shape[0], loss=loss.item() * x.shape[0])

        return loss

    def sample_fn(self, sample_size=None, noise=None, diffusion=None, sample_seed=None):

        if noise is None:
            # shape = (sample_size // self.world_size, ) + self.shape
            shape = (sample_size // self.world_size, ) + (1, *self.shape)
        else:
            shape = noise.shape

        if diffusion is None:
            diffusion = self.diffusion

        sample = diffusion.p_sample(
            denoise_fn=self.model, shape=shape,
            device=self.device, noise=noise, seed=sample_seed)
        assert sample.grad is None

        return sample

    def train(self, evaluator=None, chkpt_path=None, image_dir=None):

        # config real images for FID score computing

        import torch.nn.functional as F
        import numpy as np
        from scipy.linalg import sqrtm
        import nibabel as nib

        start_time = time.time()

        if self.num_save_images:
            assert self.num_save_images % self.world_size == 0, "Number of samples should be divisible by WORLD_SIZE!"

        global_steps = 0

        for epoch in range(self.start_epoch, self.epochs):

            self.stats.reset()
            self.model.train()

            results = dict()

            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(epoch)

            # with tqdm(self.trainloader, desc=f"{epoch+1}/{self.epochs} epoch", leave=False) as t:

                # for i, x in enumerate(t):
            torch.autograd.set_detect_anomaly(True)
            epoch_loss = 0

            for iteration, x in enumerate(self.trainloader):

                global_steps += 1
                loss = self.step(x[0].to(self.device), global_steps=global_steps)
                epoch_loss += loss
                # t.set_postfix(self.current_stats)
                if iteration % 50 == 0:
                    print('time:' + str(time.time() - start_time) + ', epoch:' + str(epoch) + ', iteration:' + str(iteration) + ', loss:' + str(
                        self.current_stats))
                    self.writer.add_scalar('loss', loss, iteration + epoch * len(self.trainloader))

            self.scheduler.step()

            if not (epoch + 1) % self.image_intv and self.num_save_images and image_dir:
                self.model.eval()
                x = self.sample_fn(sample_size=self.num_save_images, sample_seed=self.sample_seed)
                self.writer.add_scalar('epoch loss', epoch_loss, epoch)
                nib.save(nib.Nifti1Image(x.cpu().squeeze().permute([1, 2, 3, 0]).numpy(), np.eye(4)), os.path.join(image_dir, f"{epoch + 1}"))

            if not (epoch + 1) % self.chkpt_intv and chkpt_path:
                self.model.eval()
                if evaluator is not None:
                    eval_results = evaluator.eval(self.sample_fn, is_leader=self.is_leader)
                else:
                    eval_results = dict()
                results.update(eval_results)
                self.save_checkpoint(chkpt_path, epoch=epoch + 1)

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    def load_checkpoint(self, chkpt_path, map_location):
        chkpt = torch.load(chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except RuntimeError:
                _chkpt = chkpt[trainee]["shadow"] if trainee == "ema" else chkpt[trainee]
                for k in list(_chkpt.keys()):
                    if k.startswith("module."):
                        _chkpt[k.split(".", maxsplit=1)[1]] = _chkpt.pop(k)
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, epoch):

        torch.save(self.model.state_dict(), chkpt_path +'/chkpt_' + str(epoch) + '.pkl')

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


if __name__ == '__main__':

    from torch.utils.data import TensorDataset, DataLoader
    import torch

    device = torch.device('cuda')
    # change to your own dataset
    loader = DataLoader(TensorDataset(torch.rand([12, 1, 48, 48, 48])), batch_size=6, drop_last=True, shuffle=True)

    print(len(loader))
    from models.unet import UNetModel

    model = nn.parallel.DataParallel(UNetModel(image_size=48, in_channels=1, model_channels=64, out_channels=1, channel_mult=(1, 2, 4, 8),

                                               num_res_blocks=3, attention_resolutions=[], dropout=0.2,
                                               dims=3).cuda())
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)

    betas = get_beta_schedule('linear', 0.0001, 0.02, 1000).to(device)
    diffusion = GaussianDiffusion(betas, model_mean_type='eps', model_var_type='fixed-small', loss_type='mse', )

    trainer = Trainer(model, optim, diffusion, 500, loader, scheduler=sched, device=torch.device('cuda'))
    trainer.train(chkpt_path='chkpt', image_dir='images')
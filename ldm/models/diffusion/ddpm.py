"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os
import torch
import torchvision
from PIL import Image
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
import matplotlib.pyplot as plt
from omegaconf import ListConfig
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim_integration import DDIMSampler
import random
from ternary.helpers import simplex_iterator
import pickle

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    """
    Classic DDPM with Gaussian diffusion, in image space. 
    This implementation supports conditional and unconditional diffusion models.
    """

    def __init__(self,
   
                 save_dir,
                 timesteps=1000,
                 if_ddim=False,
                 eta=0,
                 seis_config=None,
                 well_config=None,
                 geo_config=None,
                 back_config=None,
                 beta_schedule="linear",
                 ddim_steps=20,
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 cond_stage_key='image',
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 make_it_fit=False,
                 ucg_training=None,
                 reset_ema=False,
                 reset_num_ema_updates=False,
                 unconditional_guidance_scale=None,
                 factor_0=None,
                 factor_1=None,
                 factor_2=None):
        """
        Initialize DDPM model parameters.
        """
        super().__init__()

        # Ensure valid parameterization
        assert parameterization in ["eps", "x0", "v"], 'Supported parameterization are "eps", "x0", and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        # Configurations
        self.cond_stage_model = None
        all_config = self.initialize_configs(seis_config, well_config, geo_config, back_config)
        
        self.loss_type=loss_type
        
        # Model initialization
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.factor_0, self.factor_1, self.factor_2 = factor_0, factor_1, factor_2
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.conditioning_key = conditioning_key
        self.save_dir=save_dir

        # Model wrapper
        self.model = DiffusionWrapper(conditioning_key, all_config)
        self.if_ddim = if_ddim
        self.eta=eta
        self.ddim_steps = ddim_steps
        count_params(self.model, verbose=True)

        # EMA (Exponential Moving Average)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # Scheduler configuration
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        # Diffusion model parameters
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        # Checkpoint restoration
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            if reset_ema:
                assert self.use_ema
                print(f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)

        # Reset EMA updates if needed
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

        # Register diffusion schedule
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        # Log variance settings
        self.learn_logvar = learn_logvar
        logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.register_buffer('logvar', logvar)

        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

    def initialize_configs(self, seis_config, well_config, geo_config, back_config):
        """
        Initializes configuration dictionary for model.
        """
        all_config = {
            'seis_config': seis_config,
            'well_config': well_config,
            'geo_config': geo_config,
            'back_config': back_config
        }
        return all_config



    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
  


    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self,c, x, t, clip_denoised: bool):
        model_out = self.model(x, t,c)
        
        

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        
      
    
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance,model_out

    @torch.no_grad()
    def p_sample(self,c, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance,model_out = self.p_mean_variance(c=c,x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,model_out

    @torch.no_grad()
    def p_sample_loop(self,c, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        torch.manual_seed(21)
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):

            img,model_out = self.p_sample(c,img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(model_out)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, c, if_ddim, batch_size=16, return_intermediates=False, factor_0=None, factor_1=None, factor_2=None):
        """
        Generate samples from the model, using either DDIM or the normal p_sample loop.
        Args:
            c: Conditioning variable.
            if_ddim: Whether to use DDIM sampling.
            batch_size: Number of samples to generate.
            return_intermediates: Whether to return intermediate results.
            factor_0, factor_1, factor_2: Parameters controlling guidance during sampling.
        Returns:
            Generated samples and intermediates (if requested).
        """
        image_size = self.image_size
        channels = self.channels

        if if_ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (batch_size, channels, self.image_size[0], self.image_size[1])
            samples, intermediates = ddim_sampler.sample(
                batch_size, shape, self.device, cond_scale=6., conditioning=c, 
                rescaled_phi=0.7, clip_denoised=True, log_every_t=self.log_every_t,
                unconditional_guidance_scale=self.unconditional_guidance_scale,
                factor_0=factor_0, factor_1=factor_1, factor_2=factor_2
            )
            return samples, intermediates
        else:
            return self.p_sample_loop(c, (batch_size, channels, self.image_size[0], self.image_size[1]), return_intermediates=return_intermediates)
            
        

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        
        return loss

    def p_losses(self, x_start,c, noise=None):
        
       
        
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()    
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.model(x_noisy, t,c)
   

        
  
        
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        loss =self.get_loss(model_out[0], target, mean=False).mean(dim=[1, 2, 3]).mean()        
 
        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_vel': loss.mean()})
       
        return loss, loss_dict

    def forward(self, x,c, *args, **kwargs):

        return self.p_losses(x,c, *args, **kwargs)

    def get_input(self, batch, k):
        
        x = batch[k]
        if self.conditioning_key=="geo":
            x=None        
        if len(x.shape) == 3:
            x = x[..., None]
      
     
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
        
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        x_recon = self.model(x_noisy, t, cond)


        return x_recon
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        c={}
        for c_s_k in self.cond_stage_key:
  
          c[c_s_k]=self.get_input(batch, c_s_k)
          if random.randint(0,5)==0:       
            c[c_s_k]=torch.zeros_like(c[c_s_k])      
        
        loss, loss_dict = self(x,c)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.shared_step(batch)
  
            
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
      
        images=self.log_images(batch)
        loss_l2=torch.nn.functional.mse_loss(images["samples"], images["inputs"])
        
      
        self.log("L2loss:", loss_l2,
                 prog_bar=False, logger=True, on_step=False, on_epoch=True)
      

        self.log_local(images,batch_idx)
        

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
    
        images=self.log_images(batch)
        
        self.log_local(images,batch_idx)
        



    def log_local(self, images, batch_idx):

        root = os.path.join(self.save_dir, "image_log_val")
        for k in images:


                
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_{}_{}.png".format(k,self.current_epoch,batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                #if k not in [ "samples_factor", "denoise_row","diffusion_row"]:
                Image.fromarray(grid).save(path)
                
                #np.save(path.replace("png","npy"), images[k].cpu().numpy())

                
   
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid




    @torch.no_grad()
    def log_images(self, batch, N=2000, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        c = {}
        if self.conditioning_key is None:
            c = None  # No conditioning        
        # Get conditions for all keys
        for c_s_k in self.cond_stage_key:
            c[c_s_k] = self.get_input(batch, c_s_k).to(self.device)[:N]
            
        # Get input image
        x = self.get_input(batch, self.first_stage_key).to(self.device)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x[:N]
        c = {k: v[:N] for k, v in c.items()}
        
        log["inputs"] = x
        for c_s_k in self.cond_stage_key:
            if c_s_k != "seis":
              log[c_s_k]=c[c_s_k]    
        # Diffusion row
        diffusion_row = []
        x_start = x[:n_row]

       
            
        for t in range(self.num_timesteps):
            log_every_t = 50
            if t % log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row).to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)
    
        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)
    
        if sample:
            # Get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(c, self.if_ddim, batch_size=N, return_intermediates=True, 
                                                   )
            
            log["samples"] = samples
            log["gap"] = log["inputs"] - log["samples"]
            log["denoise_row"] = self._get_rows_from_list(denoise_row['pred_x0'] if self.if_ddim else denoise_row)
    
        if return_keys:
            return {key: log[key] for key in return_keys if key in log}
    
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
    
        params = list(self.model.parameters())
       
       
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
 
  


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, conditioning_key, all_config):
        """
        Initializes the DiffusionWrapper module, instantiating different diffusion models based on the 
        given conditioning key.
        
        Args:
            conditioning_key (str): A key indicating which conditioning models to use (e.g., 'seis', 'well', 'geo', 'back').
            all_config (dict): A dictionary containing configuration details for each model (seis_config, well_config, etc.).
        """
        super().__init__()

        self.conditioning_key = conditioning_key

        # Conditional model instantiations based on the conditioning key
        if 'seis' in self.conditioning_key:
            self.diffusion_model_seis = instantiate_from_config(all_config["seis_config"])

        if 'well' in self.conditioning_key:
            self.diffusion_model_well = instantiate_from_config(all_config["well_config"])

        if 'geo' in self.conditioning_key:
            self.diffusion_model_geo = instantiate_from_config(all_config["geo_config"])

        if 'back' in self.conditioning_key:
            self.diffusion_model_back = instantiate_from_config(all_config["back_config"])

    def forward(self, x, t, c):
        """
        Forward pass through the model, using the conditioning input and the time step.

        Args:
            x (torch.Tensor): The noisy input tensor at time step t.
            t (torch.Tensor): The current time step.
            c (dict): A dictionary containing conditioning information (e.g., 'seis', 'well', 'geo', 'back').

        Returns:
            List[torch.Tensor]: A list of model outputs, corresponding to the diffusion models used.
        """
        out = []

        # Process each conditioning model if the corresponding key is present in the conditioning_key
        if 'seis' in self.conditioning_key:
            out.append(self.diffusion_model_seis(c["seis"], x, t))

        if 'well' in self.conditioning_key:
            xc = torch.cat([x, c["well"]], dim=1)
            out.append(self.diffusion_model_well(xc, t))

        if 'geo' in self.conditioning_key:
            out.append(self.diffusion_model_geo(x, t))

        if 'back' in self.conditioning_key:
            xc = torch.cat([x, c["back"]], dim=1)
            out.append(self.diffusion_model_back(xc, t))

        return out

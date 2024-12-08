"""SAMPLING ONLY."""
import cv2 as cv
from einops import rearrange
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor






class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.device=model.device
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.make_schedule(ddim_num_steps=model.ddim_steps, ddim_eta=model.eta, verbose=False)
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor: 
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
    @torch.no_grad()
    def sample(self,
               batch_size,
               shape,
               device,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               x_T=None,
               log_every_t=None,
               unconditional_guidance_scale=None,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               factor_0=None,
               factor_1=None,
               factor_2=None,
               **kwargs
               ):

        
        # sampling
        C, H, W = shape[1:]
        size = (batch_size, C, H, W)
       
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')


        samples, intermediates= self.ddim_sampling(conditioning, size, device,
                                                      callback=callback,
                                                      img_callback=img_callback,
                                                      quantize_denoised=quantize_x0,
                                                      mask=mask, x0=x0,
                                                      ddim_use_original_steps=False,
                                                      noise_dropout=noise_dropout,
                                                      temperature=temperature,
                                                      score_corrector=score_corrector,
                                                      corrector_kwargs=corrector_kwargs,
                                                      x_T=x_T,
                                                      log_every_t=log_every_t,
                                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                                      unconditional_conditioning=unconditional_conditioning,
                                                      dynamic_threshold=dynamic_threshold,
                                                      ucg_schedule=ucg_schedule,
                                                      )
                                                
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, c, shape, device,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=None,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None,repeat_noise=False):

        
        b = shape[0]
        with torch.random.fork_rng():
            torch.manual_seed(42)        
            if x_T is None:
                img = torch.randn(shape, device=device)
            else:
                img = x_T
       
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        

        
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            
            t = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

              
                      
     
            

            model_output_ori = self.model.apply_model(img, t, c)
           
            if len(model_output_ori)==1:
            
             model_output = model_output_ori[0]
             
            elif len(model_output_ori)==2:
             
                          
             model_output = self.model.factor_1 * model_output_ori[1] + self.model.factor_0 * model_output_ori[0]
             
            elif len(model_output_ori)==3:
            
             model_output = self.model.factor_0 * model_output_ori[0] +self.model.factor_1* model_output_ori[1]+ self.model.factor_2  *model_output_ori[2]
             
            else:
              print("The length of the model output is zero!")
              
              exit(1)  

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(img, t, model_output)
            elif self.model.parameterization == "x0":
               
                e_t=  self._predict_eps_from_xstart(img, t, model_output)
            else:
                e_t = model_output
    
            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, img, t, c, **corrector_kwargs)
                
            alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if ddim_use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if ddim_use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
    
            # current prediction for x_0
            if self.model.parameterization =='eps':
               
                pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            elif self.model.parameterization =='x0':
                pred_x0 = model_output
    
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
    
            if dynamic_threshold is not None:
                raise NotImplementedError()
            
            pred_x0.clamp_(-1., 1.) 
            
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
            img= a_prev.sqrt() * pred_x0 + dir_xt + noise
            
            
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
      
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                
        
        return pred_x0, intermediates      
         
    @torch.no_grad()    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
      return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
           extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
     
    

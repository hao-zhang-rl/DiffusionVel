import torch
import time
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ldm.models.diffusion.utils import *
import cv2
import os
import transforms as T
os.makedirs("figures", exist_ok=True)

def fwi_gradient(vel,init,device):
    dev = device# torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # configure
    model_scale = 1 # 1/2
    expand = -10
    expand = int(expand/model_scale)
    delay = 150 # ms
    fm = 5 # Hz
    dt = 0.001 # s
    nt = 1000 # timesteps
    dh = 20 # m
    pmln = 50
    srcz = 5+pmln # grid point
    recz = 5+pmln # grid point
    
    # Training
    criterion = torch.nn.MSELoss()
    lr = 20
    epochs = 20
    
    # Load velocity
    #vel = vel.cpu().detach().numpy().squeeze()#np.load("/home/haozhang/work/dataset/curve_fault_b/CurveFault_B//vel7_1_17.npy")[0,0,:,:]
    min_val=1500;max_val=4500
    init=init.clone()
    vel=np.squeeze(T.tonumpy_denormalize(vel, min_val, max_val, exp=False))
    init=np.squeeze(T.tonumpy_denormalize(init, min_val, max_val, exp=False))
    #print(init.shape)
    #print(vel.shape)    
    vel = cv2.resize(vel, (70 * 1, 70), interpolation=cv2.INTER_LINEAR)
    init= cv2.resize(init, (70 * 1, 70), interpolation=cv2.INTER_LINEAR)
    #cv2.GaussianBlur(vel, (25,25), 15)   
    #init[:, :] = init[:, 30][:, np.newaxis]

    
    #init = np.load("linear_vp.npy")

    #exit(1)
    vel = vel[::model_scale,::model_scale]
    init = init[::model_scale,::model_scale]
    vel = np.pad(vel, ((pmln, pmln), (pmln, pmln)), mode="edge")
    init = np.pad(init, ((pmln, pmln), (pmln, pmln)), mode="edge")
    pmlc = generate_pml_coefficients_2d(vel.shape, N=pmln, multiple=False)
    
    
    domain = vel.shape
    nz, nx = domain
    # imshow(vel, vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 4))
    
    # load wave
    wave = ricker(np.arange(nt) * dt-delay*dt, f=fm)
    tt = np.arange(nt) * dt
    #plt.plot(tt, wave.cpu().numpy())
    #plt.title("Wavelet")
    #plt.show()
    # Frequency spectrum
    # Show freq < 10Hz
    freqs = np.fft.fftfreq(nt, dt)[:nt//2]
    amp = np.abs(np.fft.fft(wave.cpu().numpy()))[:nt//2]
    amp = amp[freqs <= 20] 
    freqs = freqs[freqs <= 20]
    #plt.plot(freqs, amp)
    #plt.title("Frequency spectrum")
    #plt.show()
    # Geometry
    srcxs = np.arange(expand+pmln, nx-expand-pmln, 1).tolist()
    
    srczs = (np.ones_like(srcxs) * srcz).tolist()
    src_loc = list(zip(srcxs, srczs))
    #print(src_loc)
    recxs = np.arange(expand+pmln, nx-expand-pmln, 1).tolist()
    reczs = (np.ones_like(recxs) * recz).tolist()
    rec_loc = list(zip(recxs, reczs))
    
    # show geometry
    #showgeom(vel, src_loc, rec_loc, figsize=(5, 4))
    print(f"The number of sources: {len(src_loc)}")
    print(f"The number of receivers: {len(rec_loc)}")
    
    # forward for observed data
    # To GPU
    vel = torch.from_numpy(vel).float().to(dev)
    start_time = time.time()
    with torch.no_grad():
        rec_obs = forward(wave, vel, pmlc, np.array(src_loc), domain, dt, dh, dev, recz, pmln)
    end_time = time.time()
    print(f"Forward modeling time: {end_time - start_time:.2f}s")
    # Show gathers
    #show_gathers(rec_obs.cpu().numpy(), figsize=(10, 6))
    
    
    # forward for initial data
    # To GPU
    init = torch.from_numpy(init).float().to(dev)
    init.requires_grad = True
    # Configures for training
    opt = torch.optim.Adam([init], lr=lr)
    
    def closure():
        opt.zero_grad()
        #rand_size = 8
        #rand_shots = np.random.randint(0, len(src_loc), size=rand_size).tolist()
        rec_syn = forward(wave, init, pmlc, np.array(src_loc), domain, dt, dh, dev, recz, pmln)
        loss = criterion(rec_syn, rec_obs)
        loss.backward()
        return loss
    Loss = []
    #imshow(vel.cpu().detach().numpy()[pmln:-pmln,pmln:-pmln], vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 3), savepath=f"figures/true.png")
    for epoch in tqdm.trange(epochs):
        loss = opt.step(closure)
        Loss.append(loss.item())
        if epoch % (epochs-1) == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            #imshow(init.cpu().detach().numpy()[pmln:-pmln,pmln:-pmln], vmin=1500, vmax=5500, cmap="seismic", figsize=(5, 3), savepath=f"figures/{epoch:03d}.png")
            #plt.show()
    #plt.plot(Loss)
    #plt.xlabel("Epoch")
    #plt.ylabel("Loss")
    #plt.show()
    init=init[pmln:-pmln,pmln:-pmln]
    return  T.minmax_normalize(init.unsqueeze(0).unsqueeze(0), min_val, max_val)
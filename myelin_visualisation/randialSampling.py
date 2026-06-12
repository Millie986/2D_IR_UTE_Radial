import math
import torch
import numpy as np
import torchkbnufft as tkbn
import matplotlib.pyplot as plt


class RadialSampling():
    def __init__(self, img_size):
        self.img_size = img_size
        self.spokelength = self.img_size
        self.grid_size = (self.spokelength, self.spokelength)
        self.nspokes = int(np.ceil(self.spokelength * math.pi) + 5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nufft_ob = tkbn.KbNufft(im_size=(img_size, img_size), grid_size= self.grid_size).to(self.device)
        self.adjnufft_ob = tkbn.KbNufftAdjoint(im_size=(img_size, img_size), grid_size=self.grid_size).to(self.device)
    

    def kx_ky_coordinates(self):
        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
        kx = np.zeros(shape=(self.spokelength, self.nspokes))
        ky = np.zeros(shape=(self.spokelength, self.nspokes))
        ky[:, 0] = np.linspace(0, np.pi, self.spokelength)
        for i in range(1, self.nspokes):
            kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
            ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
        self.ky = np.transpose(ky)
        self.kx = np.transpose(kx)
    


    def get_kdata_ktraj(self, df, mag_readout_start, prob_map, n_readout, t2, t_readout):
        """
        Call this function to acquire the full kdata and ktraj through doing radial sampling 
        with n_readout within single inversion time
        """
        self.kx_ky_coordinates()
        self.n_on_resonance = int(len(df) / 2)

        kdata_total, ktraj_total = [], []
        total_num = int(np.ceil(self.nspokes/n_readout))
        groups = [list(range(i, min(i+n_readout, self.nspokes))) for i in range(0, self.nspokes, n_readout)]

        for i in range(total_num):
            gp = groups[i]
            for j, mag in enumerate(mag_readout_start):
                if j >= len(gp):
                    break
                magnetisation_xy = np.sqrt(mag[0]**2 + mag[1]**2)
                signal = np.full((self.img_size, self.img_size), magnetisation_xy, dtype=np.float64)
                signal_weighted = signal * prob_map

                ky = self.ky[gp[j], :]
                kx = self.kx[gp[j], :]
                ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
                ktraj_total.append(ktraj)
                signal_weighted = torch.as_tensor(signal_weighted, device= self.device).to(torch.complex128)    
                if signal_weighted.ndim == 2:
                    signal_weighted = signal_weighted[None, None, ...] 
                ktraj = torch.as_tensor(ktraj, device= self.device).to(torch.float64)

                with torch.no_grad():                            # forward (turn off gradient when no need, save memory)
                    kdata = self.nufft_ob(signal_weighted, ktraj)
                kdata_total.append(kdata)
        
        kdata_total = torch.cat(kdata_total, dim=2)
        ktraj_total = [torch.as_tensor(x, device=self.device).to(torch.float64) for x in ktraj_total]
        ktraj_total = torch.cat(ktraj_total, dim=1)
        
        t = np.linspace(0, t_readout, self.spokelength)
        kdata_numpy = np.reshape(kdata_total.cpu().numpy(), (self.nspokes, self.spokelength)) * np.exp(-t/t2)
        kdata_t = torch.from_numpy(kdata_numpy).to(torch.complex128).reshape(1, 1, -1)

        # siglevel = torch.abs(kdata_t).mean()
        # noise = torch.randn(kdata_t.shape) + 1j * torch.randn(kdata_t.shape)
        # noise = noise.to(kdata_t)
        # kdata_t = kdata_t + (siglevel/5) * noise
        sigma = 1
        noise = (sigma / (2 ** 0.5)) * (torch.randn_like(kdata_t.real) + 1j * torch.randn_like(kdata_t.real))
        kdata_t = kdata_t + noise
        return kdata_t, ktraj_total
    


    def visualReadoutsInSingleTI(self, mag_readout_start, df, prob_map):
        self.kx_ky_coordinates()
        self.n_on_resonance = int(len(df) / 2)
        n = len(mag_readout_start)
        ncols = 5
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).ravel()
        for i, mag in enumerate(mag_readout_start):
            magnetisation_xy = np.sqrt(mag[0]**2 + mag[1]**2)
            signal = np.full(
                (self.img_size, self.img_size),
                magnetisation_xy,
                dtype=np.float64)
            signal_weighted = signal * prob_map

            # plot single graph
            ax = axes[i]
            im = ax.imshow(
                signal_weighted,
                cmap='gray',
                vmin=0,
                vmax=1)
            ax.set_title(f"Readout {i+1} | Mxy = {magnetisation_xy:.3f}", fontsize=14)
            ax.set_xlabel("x", fontsize=14)
            ax.set_ylabel("y", fontsize=14)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # close all empty subplots
        for j in range(len(mag_readout_start), len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"Mxy across {n} readouts in a single inversion time", fontsize = 16)
        plt.tight_layout()
        plt.show()


    def visual_logKdata(self, kdata, t2, t_readout):
        t = np.linspace(0,t_readout, self.spokelength)
        kdata_numpy = np.reshape(kdata.cpu().numpy(), (self.nspokes, self.spokelength)) * np.exp(-t/t2)
        plt.imshow(np.log10(np.absolute(kdata_numpy)))
        plt.gray()
        plt.title('k-space data, log10 scale')
        plt.show()

    def visual_blurry_sharp_image(self, kdata, ktraj_total):
        # method 1: no density compensation (blurry image)
        image_blurry = self.adjnufft_ob(kdata, ktraj_total)
        # method 2: use density compensation
        dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj_total, im_size=(self.img_size, self.img_size))
        image_sharp = self.adjnufft_ob(kdata * dcomp, ktraj_total)
        # show the images
        image_blurry_numpy = np.squeeze(image_blurry.cpu().numpy())
        image_sharp_numpy = np.squeeze(image_sharp.cpu().numpy())
        plt.figure(0)
        plt.imshow(np.absolute(image_blurry_numpy))
        plt.gray()
        plt.title('blurry image')
        plt.figure(1)
        plt.imshow(np.absolute(image_sharp_numpy))
        plt.gray()
        plt.title('sharp image (with Pipe dcomp)')
        plt.show()

    

       

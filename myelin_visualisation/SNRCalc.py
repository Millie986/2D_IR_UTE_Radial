import math
import torch
import numpy as np
import IR_UTE as ute
import phantom as pt
import torchkbnufft as tkbn
import randialSampling as rs
import matplotlib.pyplot as plt

class SNRCalc():
    def __init__(self, dt, df, t1_myelin, t2_myelin, t1_wm, t2_wm, img_size = 256, n_readout = 9, flip_angle = math.pi/2):
        self.dt = dt
        self.df = df
        self.t1_myelin = t1_myelin
        self.t2_myelin = t2_myelin
        self.t1_wm = t1_wm
        self.t2_wm = t2_wm
        self.img_size = img_size
        self.n_readout = n_readout
        self.flip_angle = flip_angle
        pass

    def combined_kdata_ktraj(self, t_readout, prob_map_myelin, prob_map_wm):
        print('white matter')
        wm = ute.IR_UTE(self.dt, self.df, self.t1_wm, self.t2_wm)
        self.TR = wm.TR
        _ = wm.inversion_recovery()
        t_nullout = wm.calculate_nulling_point()
        _, _, mag_readout_start_wm, _ = wm.transverse_excitation(self.n_readout, t_readout, t_nullout, self.flip_angle)

        print()
        print('myelin')
        myelin = ute.IR_UTE(self.dt, self.df, self.t1_myelin, self.t2_myelin)
        _ = myelin.inversion_recovery()
        _, _, mag_readout_start_myelin, _ = myelin.transverse_excitation(self.n_readout, t_readout, t_nullout, self.flip_angle)


        radial_sampling = rs.RadialSampling(self.img_size)
        self.nspokes = radial_sampling.nspokes
        kdata_t_myelin, ktraj_total = radial_sampling.get_kdata_ktraj(self.df, mag_readout_start_myelin, prob_map_myelin, self.n_readout, self.t2_myelin, t_readout)
        kdata_t_wm, _ = radial_sampling.get_kdata_ktraj(self.df, mag_readout_start_wm, prob_map_wm, self.n_readout, self.t2_wm, t_readout)
        
        kdata_total = kdata_t_myelin + kdata_t_wm
        self.radial_sampling = radial_sampling


        print("flip_angle:", self.flip_angle)
        print("mag_readout_start_myelin:", mag_readout_start_myelin)
        print("mag_readout_start_wm:", mag_readout_start_wm)
        return kdata_total, ktraj_total
    
    def SNR_CNR_calculation(self, kdata_total, ktraj_total, prob_map_myelin, prob_map_wm, blur=False):
        if blur:
            # method 1: no density compensation (blurry image)
            image = self.radial_sampling.adjnufft_ob(kdata_total, ktraj_total)
        else:
            # method 2: use density compensation
            dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj_total, im_size=(self.img_size, self.img_size))
            image = self.radial_sampling.adjnufft_ob(kdata_total * dcomp, ktraj_total)

    
        image_numpy = np.squeeze(image.cpu().numpy())
        print(image_numpy.shape)
        image_abs = np.abs(image_numpy) 

        noise_std = np.std(image_abs[(prob_map_myelin == 0) & (prob_map_wm == 0)])
        print(image_abs[(prob_map_myelin == 0) & (prob_map_wm == 0)])
        myelin_mean = np.mean(image_abs[prob_map_myelin == 1])
        wm_mean = np.mean(image_abs[prob_map_wm == 1])

        print('noise_std:', noise_std)
        print('noise mean:', np.mean(image_abs[(prob_map_myelin == 0) & (prob_map_wm == 0)]))
        print('myelin_mean:', myelin_mean)
        print('wm_mean:', wm_mean)
        
        snr_myelin = 0.66 * myelin_mean / noise_std
        snr_wm = 0.66 * wm_mean / noise_std
        cnr = 0.66 * abs(myelin_mean - wm_mean) / noise_std
        print("SNR_myelin:", snr_myelin)
        print("SNR_wm:", snr_wm)
        print('SNR Noise:', 0.66 * np.mean(image_abs[(prob_map_myelin == 0) & (prob_map_wm == 0)])/noise_std)
        return  snr_myelin, snr_wm, cnr
    
    def total_acquisation_time_calc(self, TR):
        """
        TR: repetition time in ms
        """
        radial_sampling = rs.RadialSampling(self.img_size)
        nspokes = radial_sampling.nspokes

        n_TR = np.ceil(nspokes / self.n_readout)
        total_TA = n_TR * TR
        return total_TA



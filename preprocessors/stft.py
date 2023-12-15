import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats

def _first(arr, axis):
    #from https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L2662-L2730
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position."""
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)

def zscore(a, axis):
    #from https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L2662-L2730
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)

    std[(std==0)] = 1.0 #this is a hack. I should eventually find where the bad data is
    z = (a - mn) / std
    return z, mn, std

# TODO: Reorganize this to return a dictionary
class STFTPreprocessor(nn.Module):
    def get_stft(self, x, fs, show_fs=-1, normalizing=None, **kwargs):
        # print(x, fs, kwargs)
        f, t, Zxx = signal.stft(x, fs, **kwargs)
        # print(f, t, Zxx)
        
        # print("1 f shape: {}, t shape: {}, Zxx shape: {}".format(f.shape, t.shape, Zxx.shape))

        if "return_onesided" in kwargs and kwargs["return_onesided"] == True:
            erased_Zxx = np.abs(Zxx[show_fs:, :]).copy()
            Zxx = Zxx[:show_fs, :].copy()
            f = f[:show_fs]
        else:
            pass #TODO
            #Zxx = np.concatenate([Zxx[:,:,:show_fs], Zxx[:,:,-show_fs:]], axis=-1)
            #f = np.concatenate([f[:show_fs], f[-show_fs:]], axis=-1)

        # print("2 f shape: {}, t shape: {}, Zxx shape: {}".format(f.shape, t.shape, Zxx.shape))

        Zxx = np.abs(Zxx)
        
        un_normalized_Zxx = Zxx.copy()

        if normalizing=="zscore":
            Zxx, mn, stf = zscore(Zxx, axis=-1)#TODO is this order correct? I put it this way to prevent input nans
            if (Zxx.std() == 0).any():
                Zxx = np.ones_like(Zxx)
            Zxx = Zxx[:,10:-10]
            t = t[10:-10]
            un_normalized_Zxx = un_normalized_Zxx[:,10:-10]
            erased_Zxx = erased_Zxx[:,10:-10]
        elif normalizing=="db":
            Zxx = np.log(Zxx)

        if np.isnan(Zxx).any():
            Zxx = np.nan_to_num(Zxx, nan=0.0)
            
        # print("3 f shape: {}, t shape: {}, Zxx shape: {}".format(f.shape, t.shape, Zxx.shape))
        if normalizing=="zscore":
            return f, t, torch.Tensor(np.transpose(Zxx)), mn, stf, np.transpose(un_normalized_Zxx), erased_Zxx
        else:
            return f, t, torch.Tensor(np.transpose(Zxx)), np.transpose(un_normalized_Zxx), erased_Zxx

    def __init__(self, cfg):
        super(STFTPreprocessor, self).__init__()
        self.cfg = cfg

    def forward(self, wav):
        f, t, linear, mn, stf, un_normalized_Zxx, erased_Zxx = self.get_stft(wav, self.cfg.fs, show_fs=self.cfg.freq_channel_cutoff, nperseg=self.cfg.nperseg, noverlap=self.cfg.noverlap, normalizing=self.cfg.normalizing, return_onesided=True) #TODO hardcode sampling rate
        # _,_,linear = self.get_stft(wav, 1000, show_fs=self.cfg.freq_channel_cutoff, nperseg=self.cfg.nperseg, noverlap=self.cfg.noverlap, normalizing=self.cfg.normalizing) #TODO hardcode sampling rate
        return f, t, linear, mn, stf, un_normalized_Zxx, erased_Zxx

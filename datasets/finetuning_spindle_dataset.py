from omegaconf import OmegaConf
import torch
import random
from torch.utils import data 
import os
import numpy as np
from scipy.io import wavfile
from datasets import register_dataset
from preprocessors import STFTPreprocessor
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

@register_dataset(name="spindle_finetuning")
class SpindleFinetuning(data.Dataset):
    def __init__(self, cfg, task_cfg=None, preprocessor_cfg=None):
        self.cfg = cfg
        self.task_cfg = task_cfg
        manifest_path = cfg.data
        manifest_path = os.path.join(manifest_path, cfg.manifest_name)
        with open(manifest_path, "r") as f:
            lines = f.readlines() 
        self.root_dir = lines[0].strip()
        files, lengths = [], []
        for x in lines[1:]:
            row = x.strip().split('\t')
            files.append(row[0])
            lengths.append(row[1])
        self.files, self.lengths = files, lengths
        self.labels = [int("non" not in x) for x in files]

        # subsample for balanced dataset
        # rus = RandomUnderSampler(random_state=0, replacement=True)
        rus = RandomOverSampler(random_state=0)
        self.files, self.labels = rus.fit_resample(np.array(self.files).reshape(-1, 1), self.labels)
        self.files = self.files.flatten().tolist()

        self.cached_features = None

        if 'cached_features' in cfg:
            self.cached_features = cfg.cached_features
            self.initialize_cached_features(cfg.cached_features)
        elif preprocessor_cfg.name=="stft":
            extracter = STFTPreprocessor(preprocessor_cfg)
            self.extracter = extracter
        else:
            raise RuntimeError("Specify preprocessor")

    def initialize_cached_features(self, cache_root):
        cfg_path = os.path.join(cache_root, "config.yaml")
        loaded = OmegaConf.load(cfg_path)
        assert self.cfg.preprocessor == loaded.data.preprocessor

        manifest_path = os.path.join(cache_root, "manifest.tsv")
        with open(manifest_path, "r") as f:
            lines = f.readlines() 
        self.cache_root_dir = lines[0].strip()
        orig2cached = {} #Map original file to cached feature file
        for x in lines[2:]:
            row = x.strip().split('\t')
            orig2cached[row[0]] = row[1]
        self.orig2cached = orig2cached

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def __len__(self):
        return len(self.files)

    def get_cached_features(self, file_name):
        file_name = self.orig2cached[file_name] 
        file_name = os.path.join(self.cache_root_dir, file_name)
        data = np.load(file_name)
        data = np.nan_to_num(data) #For superlet caches
        data = torch.FloatTensor(data)
        return data
        
    def __getitem__(self, idx):
        file_name = self.files[idx]
        label = self.labels[idx]
        # print(file_name, label)
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path)
        
        # print("data shape: {}".format(data.shape))

        data = data.astype('float32')
        #rand_len = random.randrange(1000, len(data), 1)
        # rand_len = -1
        # wav = np.squeeze(data)[:rand_len]
        wav = np.squeeze(data)
        
        # print("wav shape: {}".format(wav.shape))

        # if self.cached_features:
        # data = self.get_cached_features(file_name)
        # else:
        f, t, data, mn, stf, un_normalized_data, erased_Zxx = self.extracter(wav)

        # only take 0-40Hz
        data[9:, :] = 0.0

        return {
            "input": data,
            "length": len(data),
            "wav": wav,
            "label": label
        }
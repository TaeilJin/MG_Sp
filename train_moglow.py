"""Train script.

Usage:
    train_moglow.py <hparams> <dataset>
"""
import os
import motion
import numpy as np
import datetime

from docopt import docopt
from torch.utils.data import DataLoader, Dataset
from glow.builder import build
from glow.trainer import Trainer
from glow.generator import Generator
from glow.config import JsonConfig
from torch.utils.data import DataLoader
import torch
if __name__ == "__main__":
    # args = docopt(__doc__)
    # hparams = args["<hparams>"]
    # dataset = args["<dataset>"]
    hparams = "hparams/preferred/locomotion.json" # 
    dataset = "locomotion"
    assert dataset in motion.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, motion.Datasets.keys()))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)
    dataset = motion.Datasets[dataset]
    
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
		
    print("log_dir:" + str(log_dir))
    print("mG_Sp model")
    is_training = hparams.Infer.pre_trained == ""
    
    data = dataset(hparams, is_training)
    x_channels, cond_channels = data.n_channels()

    # build graph
    built = build(x_channels, cond_channels, hparams, is_training)
    print("trajectory condition is 3")
    if is_training:
        # build trainer
        trainer = Trainer(**built, data=data, log_dir=log_dir, hparams=hparams)
        
        # train model
        trainer.train()
    else:
        # Synthesize a lot of data. 
        generator = Generator(data, built['data_device'], log_dir, hparams)
        if "temperature" in hparams.Infer:
            temp = hparams.Infer.temperature
        else:
            temp = 1

        # origin_apd = T1_apd = np.load('../data/results/test_moGlow/0_sampled_temp100_0k_APD_score.npz')['clips'].astype(np.float32)
        # mean_origin_apd = np.mean(origin_apd)
        # T1_apd = np.load('../data/results/test_moGlow_T1/0_sampled_temp100_0k_APD_score.npz')['clips'].astype(np.float32) 
        # mean_origin_T1_apd = np.mean(T1_apd)  
        
        # We generate x times to get some different variations for each input
        torch.manual_seed(42)
        with torch.no_grad():
            generator.generate_APD_perBatch(built['graph'],eps_std=temp,counter=0)
            # for i in range(5):            
            #     generator.generate_sample_withRef(built['graph'],eps_std=temp, counter=i)
            


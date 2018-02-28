#! -*- coding:utf-8 -*-

class NetConfig():
    def __init__(self):
        self.inference_num_units = 64
        self.inference_num_layers = 1
        self.prediction_num_units = 64
        self.prediction_num_layers = 1
        self.latent_dim = 10
        self.regularize_const = 0.5
        
        self._int = ["inference_num_units",
                     "inference_num_layers",
                     "prediction_num_units",
                     "prediction_num_layers",
                     "latent_dim"]
        self._float = ["regularize_const"]
        
    def _setattr(self, name, value):
        if name in self._int:
            value = int(value)
            setattr(self, name, value)
        elif name in self._float:
            value = float(value)
            setattr(self, name, value)
        else:
            print "{} can not be changed".format(name)
                
    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)
        
    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]

class TrainConfig():
    def __init__(self):
        self.seed = None
        self.test = 0
        self.epoch = 100
        self.log_interval = 10
        self.test_interval = 10
        self.learning_rate = 0.001
        self.batchsize = 10
        self.data_file = "./data.npy"
        self.test_data_file = None
        self.save_dir = "./checkpoints"
        self.gpu_use_rate = 0.8
        
    def _setattr(self, name, value):
        if name in ["seed", "test", "epoch",
                    "log_interval", "test_interval",
                    "batchsize"]:
            value = int(value)
        if name in ["learning_rate", "gpu_use_rate"]:
            value = float(value)
        setattr(self, name, value)
            
    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)
        
    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]

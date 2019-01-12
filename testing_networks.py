# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:26:25 2019

@author: User
"""

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
  

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
       

        for i, data in enumerate(dataset):
          
            model.set_input(data)
            model.test()
            break
        break 
    print('success')
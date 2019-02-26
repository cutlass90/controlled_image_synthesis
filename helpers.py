from zipfile import ZipFile
import datetime

import os
import numpy as np
import torch
from torch import nn


def save_weights(net, save_path):
    if isinstance(net, nn.DataParallel):
        torch.save(net.module.state_dict(), save_path)
    else:
        torch.save(net.state_dict(), save_path)
    print('Weights were saved', save_path)


def load_weights(net, load_path, load_only_existing_weights=False):
    print(f'Trying to load {load_path} ...', end=' ')
    if not os.path.isfile(load_path):
        raise FileNotFoundError(load_path)
    try:
        net.load_state_dict(torch.load(load_path))
        print('Weights were loaded successfully.')
    except:
        pretrained_dict = torch.load(load_path)
        model_dict = net.state_dict()
        try:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            net.load_state_dict(pretrained_dict)
            print('Pretrained net has excessive layers; Only loading layers that are used.')
        except:
            if load_only_existing_weights:
                print('Pretrained net has fewer layers; The following are not initialized:')
                not_initialized = []
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v
                    else:
                        print(k)

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.append(k)
                print(sorted(not_initialized))
                net.load_state_dict(model_dict)
            else:
                raise ValueError('There are no weights that can be loaded.')

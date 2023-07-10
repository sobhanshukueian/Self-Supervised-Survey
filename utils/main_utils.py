import torch
import numpy as np
import glob
import os
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt

import torchvision
import torch.nn as nn

from utils.utils import LARS
# from Barlow_model import Barlow_model
# from BYOL_model import BYOLNetwork
# from BYOL_PA_model import BYOLPANetwork 
# from MOCO_model2 import MOCO
from configs import model_config
from models.moco import MOCO_MODEL
from models.moco_var import MOCO_VAR_MODEL
from models.moco_var_detached import MOCO_DVAR_MODEL
from models.simsiam import SimSiam_MODEL
from models.simsiam_var import SimSiam_VAR_MODEL
from models.simclr import SimCLR_MODEL
from models.simclr_var import SimCLR_VAR_MODEL

# from MOCOO_model2 import MOCOOOOOOO
# from MOCOO_model3 import MOCO3
# from MOCOO_model4 import MOCO4
# from MOCOO_model5 import MOCO5
# from MOCOO_model6 import MOCO6
# from MOCOO_model7 import MOCO7
# from MOCOO_model8 import MOCO8
# from MOCOO_model9 import MOCO9
 



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def get_optimizer(logger, model_parameters, conf, resume, ckpt, optimizer, lr0=0.001, momentum=0.937, weight_decay=0.00005):
    assert optimizer == 'SGD' or 'Adam' or 'LARS', 'ERROR: unknown optimizer, use SGD defaulted'
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model_parameters, lr=lr0, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model_parameters, lr=lr0, weight_decay=1e-6)
    elif optimizer == 'AdamW':
        # optim = torch.optim.AdamW(model_parameters, lr=lr0, weight_decay=weight_decay)
        optim = torch.optim.AdamW(model_parameters)
    elif optimizer == "LARS": 
        param_weights = []
        param_biases = []
        for param in model_parameters:
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optim = LARS(parameters, lr=0, weight_decay=1e-6,
                weight_decay_filter=True,
                lars_adaptation_filter=True)

    if resume:
        optim.load_state_dict(ckpt['optimizer'])

    print(f"{'optimizer:'} {type(optim).__name__}")
    conf['Optimizer'] = f"{'optimizer:'} {type(optim).__name__}"
    logger.warning(f"{'optimizer:'} {type(optim).__name__}")
    return optim, conf

# Get Model 
def get_model(name, conf, resume, save_dir="./", weights=None, device='cpu'):
    if name == "MOCO":
        model = MOCO_MODEL()
    elif name == "MOCO_VAR":
        model = MOCO_VAR_MODEL()
    elif name == "MOCO_DVAR_MODEL":
        model = MOCO_DVAR_MODEL()
    elif name == "SimSiam":
        model = SimSiam_MODEL()
    elif name == "SimSiam_VAR":
        model = SimSiam_VAR_MODEL()
    elif name == "SimCLR":
        model = SimCLR_MODEL()
    elif name == "SimCLR_VAR":
        model = SimCLR_VAR_MODEL()
    # if name == "Barlow":
    #     model = Barlow_model()
    # elif name == "byol":
    #     model = BYOLNetwork()
    # elif name == "byol-pa":
    #     model = BYOLPANetwork()

    # elif name == "MOCOO":
    #     model = ModelMoCo()
    # elif name == "MOCOO2":
    #     model = MOCOOOOOOO()
    # elif name == "MOCO3":
    #     model = MOCO3()
    # elif name == "MOCO4":
    #     model = MOCO4()
    # elif name == "MOCO5":
    #     model = MOCO5()
    # elif name == "MOCO6":
    #     model = MOCO6()
    # elif name == "MOCO7":
    #     model = MOCO7()
    # elif name == "MOCO8":
    #     model = MOCO8()
    # elif name == "MOCO9":
    #     model = MOCO9()
    elif name == "supervised":
        model = torchvision.models.resnet50(pretrained=True, num_classes=model_config["EMBEDDING_SIZE"])
    elif name == "random":
        model = torchvision.models.resnet50(pretrained=False, num_classes=model_config["EMBEDDING_SIZE"])
    else:
        assert "Unknown Network name"

    ckpt = False

    if resume:
        # Find the most recent saved checkpoint in search_dir
        checkpoint_list = glob.glob(f'{save_dir}/**/last*.pt', recursive=True)
        checkpoint_path = max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''
        assert os.path.isfile(checkpoint_path), f'the checkpoint path is not exist: {checkpoint_path}'
        print(f'Resume training from the checkpoint file :{checkpoint_path}')
        conf['resume'] = f'Resume training from the checkpoint file :{checkpoint_path}'
        ckpt = torch.load(checkpoint_path)
    elif weights:  
        print(f'Loading state_dict from {weights} for fine-tuning...')
        ckpt = torch.load(weights)

    if ckpt:
        state_dict = ckpt['model'].float().state_dict()
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
        model.load_state_dict(state_dict, strict=False)
        del state_dict, model_state_dict
        
    return model, conf, ckpt
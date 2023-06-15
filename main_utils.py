import torch
import numpy as np
import glob
import os
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt

import torchvision
import torch.nn as nn

from utils import LARS
from Barlow_model import BarlowTwins
from BYOL_model import BYOLNetwork
# from BYOL_PA_model import BYOLPANetwork 
from MOCO_model2 import MOCO
from configs import model_config
from MOCOO_model import ModelMoCo
from MOCOO_model2 import MOCOOOOOOO
from MOCOO_model3 import MOCO3
from MOCOO_model4 import MOCO4
from MOCOO_model6 import MOCO6




from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def get_optimizer(logger, model_parameters, conf, resume, ckpt, optimizer, lr0=0.001, momentum=0.937, weight_decay=0.00005, verbose=1):
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
def get_model(name, conf, resume, save_dir="./", weights=None, device='cpu', verbose=1):
    if name == "barlow":
        model = BarlowTwins()
    elif name == "byol":
        model = BYOLNetwork()
    # elif name == "byol-pa":
    #     model = BYOLPANetwork()
    elif name == "MOCO":
        model = MOCO()
    elif name == "MOCOO":
        model = ModelMoCo()
    elif name == "MOCOO2":
        model = MOCOOOOOOO()
    elif name == "MOCO3":
        model = MOCO3()
    elif name == "MOCO4":
        model = MOCO4()
    elif name == "MOCO5":
        model = MOCO5()
    elif name == "MOCO6":
        model = MOCO6()
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
    # Log Model
    # if verbose > 0:
    #     print('Model: {}'.format(model))
    conf["Model"] = str(model)
    return model, conf, ckpt

def eval_knn(embeddings, labels, knns):
    X_train = np.array(embeddings)
    y_train = np.array(labels)

    k = 1
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(X_train, y_train)
    acc = 100 * np.mean(cross_val_score(knn, X_train, y_train))

    print(f'KNN Accuracy: {acc}')
    # print(len(knns))

    if len(knns) > 0:
        indices = range(len(knns))
        plt.figure(figsize=(10,6))
        plt.plot(indices, knns, label='train set KNN acc (%)')
        plt.ylabel('Accuracy in %')
        plt.xlabel('Epochs')
        plt.legend()
        if model_config["SAVE_PLOTS"]:
            save_plot_dir = osp.join(model_config["SAVE_DIR"], 'plots') 
            if not osp.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
            plt.savefig("{}/knn.png".format(save_plot_dir)) 
        if model_config["VISUALIZE_PLOTS"]:
            plt.show()
    return acc
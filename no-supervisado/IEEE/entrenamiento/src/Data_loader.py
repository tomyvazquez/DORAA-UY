import pandapower as pp
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_net(red,red_path=None,device="cuda"):
    if red == '30':
        net = pp.networks.case30()
        z_trafos = net.trafo[['hv_bus', 'lv_bus']].to_numpy().astype(np.int32)
    elif red == '118':
        net = pp.networks.case118()
        z_trafos =  np.array([
            [0, 0.0267],
            [0, 0.0382],
            [0, 0.0388],
            [0, 0.0375],
            [0, 0.0386],
            [0, 0.0268],
            [0, 0.0370],
            [0.0013, 0.016],
            [0, 0.0370],
            [0.0017, 0.0202],
            [0, 0.0370],
            [0.0282, 0.2074],
            [0.0003, 0.00405]
        ])
    elif red == 'uru':
        net = pp.from_pickle(red_path)
        # r = net.trafo['vkr_percent'].to_numpy() / 100 * net.sn_mva / net.trafo['sn_mva'].to_numpy()
        r = net.trafo['vkr_percent'].to_numpy() / 100 * 100 / net.trafo['sn_mva'].to_numpy()
        # z = net.trafo['vkr_percent'].to_numpy() / 100 * net.sn_mva / net.trafo['sn_mva'].to_numpy()
        z = net.trafo['vk_percent'].to_numpy() / 100 * 100 / net.trafo['sn_mva'].to_numpy()
        x = np.sqrt(z**2 - r**2)
        z_trafos = np.stack((r,x),axis=1)

    num_nodes = len(net.bus)
    num_gens = len(net.gen) + len(net.ext_grid)
    
    line_index = torch.Tensor([net.line["from_bus"],net.line["to_bus"]]).type(torch.int64).to(device)
    trafo_index = torch.Tensor([net.trafo["hv_bus"],net.trafo["lv_bus"]]).type(torch.int64).to(device)
    shunt_index =torch.Tensor(net.shunt["bus"]).type(torch.int64).to(device)
    edge_index = torch.hstack([line_index,trafo_index])
    edge_index_T = edge_index.clone()
    edge_index_T[1, :] = edge_index[0, :]
    edge_index_T[0, :] = edge_index[1, :]
    edge_index = torch.cat((edge_index, edge_index_T), dim=1)

    # Armar matriz de pesos
    k = 10
    z_lineas =  net.line[['r_ohm_per_km', 'x_ohm_per_km']].to_numpy() * 100 / np.expand_dims((net.bus['vn_kv'][net.line['from_bus'].to_numpy()].to_numpy())**2,axis=1)
    edge_weights = np.append(z_lineas, z_trafos,axis=0)
    edge_weights = torch.Tensor(np.e**(-k*(edge_weights[:,0]**2 + edge_weights[:,1]**2))).to(device)
    edge_weights = torch.cat((edge_weights, edge_weights), dim=0)

    # feature_mask = np.zeros(len(net.bus.index), dtype=int)
    # feature_mask[ids] = 1
    # feature_mask = torch.Tensor(feature_mask).type(torch.int32).to(device)

    # print("edge_index",edge_index)
    # print("edge_weights",len(edge_weights))

    return edge_index, edge_weights, net




def load_data(data_path, batch_size, normalize_X, red, device):
    

    X_tensor_train = (torch.Tensor(np.load(data_path+f'/red{red}/train/input.npy') ) / 100).to(device)        
    X_tensor_val = (torch.Tensor(np.load(data_path+f'/red{red}/val/input.npy') ) / 100).to(device)        
    X_tensor_test = (torch.Tensor(np.load(data_path+f'/red{red}/test/input.npy') ) / 100).to(device)        

    norm_x = None
    # Normalizar X
    if normalize_X:
        mean = torch.mean(X_tensor_train,0)

        std = torch.std(X_tensor_train,0)
        std[std == 0.] = float('inf')
        
        X_tensor_train  = (X_tensor_train - mean) / std
        X_tensor_val  = (X_tensor_val - mean) / std
        X_tensor_test  = (X_tensor_test - mean) / std
        
        norm_x = {"mean": mean, "std": std}

    X_train = TensorDataset(X_tensor_train)
    X_val = TensorDataset(X_tensor_val)
    X_test = TensorDataset(X_tensor_test)


    train_loader = DataLoader(X_train, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(X_val, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, test_loader, norm_x
import pandapower as pp
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_net(red,device="cuda"):
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

    num_nodes = len(net.bus)
    num_gens = len(net.gen) + len(net.ext_grid)
    
    # Armar matirz de adyacencia
    lineas = net.line[['from_bus', 'to_bus']].to_numpy().astype(np.int32)
    trafos = net.trafo[['hv_bus', 'lv_bus']].to_numpy().astype(np.int32)
    edge_index = np.append(lineas, trafos,axis=0)
    edge_index = torch.Tensor(edge_index).t().type(torch.int64).to(device)

    edge_index_T = edge_index.clone()
    edge_index_T[1, :] = edge_index[0, :]
    edge_index_T[0, :] = edge_index[1, :]
    edge_index = torch.cat((edge_index, edge_index_T), dim=1)

    # Armar matriz de pesos
    k = 10
    z_lineas =  net.line[['r_ohm_per_km', 'x_ohm_per_km']].to_numpy() * 100 / np.expand_dims((net.bus['vn_kv'].to_numpy()[net.line['from_bus'].to_numpy()])**2,axis=1)
    edge_weights = np.append(z_lineas, z_trafos,axis=0)
    edge_weights = torch.Tensor(np.e**(-k*(edge_weights[:,0]**2 + edge_weights[:,1]**2))).to(device)
    edge_weights = torch.cat((edge_weights, edge_weights), dim=0)

    # Armar mascara de generadores para la salida
    idx_gen = net.gen["bus"].to_numpy()
    # idx_grid = net.ext_grid["bus"].to_numpy()
    #idx_gens = np.append(idx_gen,idx_grid,axis=0)

    feature_mask = np.zeros(len(net.bus.index), dtype=int)
    feature_mask[idx_gen] = 1
    feature_mask = torch.Tensor(feature_mask).type(torch.int32).to(device)

    return num_nodes, num_gens, edge_index, edge_weights, feature_mask, net




def load_data(data_path, batch_size, normalize_X, normalize_Y, device):
    # Levantar los datos
 
    # X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.1,random_state=42)
    X_tensor_train = (torch.Tensor(np.load(os.path.join(data_path, 'train/input.npy')))).to(device) 
    y_tensor_train = (torch.Tensor(np.load(os.path.join(data_path, 'train/vm_pu_opt.npy')))).to(device)       
    X_tensor_val = (torch.Tensor(np.load(os.path.join(data_path, 'val/input.npy')))).to(device)   
    y_tensor_val = (torch.Tensor(np.load(os.path.join(data_path, 'val/vm_pu_opt.npy')))).to(device)       
    X_tensor_test = (torch.Tensor(np.load(os.path.join(data_path, 'test/input.npy')))).to(device)        
    y_tensor_test = (torch.Tensor(np.load(os.path.join(data_path, 'test/vm_pu_opt.npy')))).to(device)


    # Normalizar X
    norm_X = None
    if normalize_X:
        mean_X = torch.mean(X_tensor_train,0)

        std_X = torch.std(X_tensor_train,0)
        std_X[std_X == 0.] = float('inf')
        
        X_tensor_train  = (X_tensor_train - mean_X) / std_X
        X_tensor_val  = (X_tensor_val - mean_X) / std_X
        X_tensor_test  = (X_tensor_test - mean_X) / std_X
        
        norm_X = (mean_X,std_X)

    norm_y = None
    if normalize_Y:
        mean_y = torch.mean(y_tensor_train,0)

        std_y = torch.std(y_tensor_train,0)
        std_y[std_y == 0.] = float('inf')
        # print('std_y', std_y)
        
        y_tensor_train  = (y_tensor_train - mean_y) / std_y
        y_tensor_val  = (y_tensor_val - mean_y) / std_y
        y_tensor_test  = (y_tensor_test - mean_y) / std_y
        
        norm_y = (mean_y,std_y)

 
    dataset_train = TensorDataset(X_tensor_train, y_tensor_train)
    dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
    dataset_test = TensorDataset(X_tensor_test, y_tensor_test)

    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)

    return train_loader, val_loader, test_loader, norm_X, norm_y
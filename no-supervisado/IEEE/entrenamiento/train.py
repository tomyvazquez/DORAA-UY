# IMPORTS
import optuna
import torch
import os
import json
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
import pandapower as pp
import networkx as nx
from torch.utils.tensorboard import SummaryWriter

from src.arquitecturas import GNNUnsupervised, FCNNUnsupervised
from src.Data_loader import load_net, load_data
from src.train_eval import run_epoch, evaluate
from src.utils import get_Ybus, get_Yline, init_lamdas, get_line
from src.Loss import my_loss, get_max_min_values
from src.metric import feas_and_volt_metric

import warnings
warnings.filterwarnings('ignore')


# OBJETIVO OPTUNA

def objective(trial):
    cfg = OmegaConf.load(args.cfg)
    
    # BUSQUEDA DE HIPERPARAMETROS
    
    ## Entrenamiento
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32])
    
    # Data
    norm_X = trial.suggest_categorical('norm_X', [True, False])
    
    ## Regularaizacion
    weight_dec = trial.suggest_loguniform('weight_decay', 0.000001, 0.01)
    drop = trial.suggest_uniform('dropout', 0, 0.5)
    
    # Penalizaciones
    if cfg.data.red == "30":    
        c_1 = trial.suggest_loguniform('c_1', 10, 1000)
        c_2 = trial.suggest_loguniform('c_2', 10, 1000)
        c_3 =  trial.suggest_loguniform('c_3', 1e-6, 1e-2)
    elif cfg.data.red == "118":
        c_1 = trial.suggest_loguniform('c_1', 1, 300)
        c_2 = trial.suggest_loguniform('c_2', 1, 300)
        c_3 =  trial.suggest_loguniform('c_3', 1e-5, 1e-1)
    else:
        print("ERROR EN LA NET")

    ## Modelo
    if cfg.model.model == "GNN":
        k = trial.suggest_int('k', 2,8)
        layers =  trial.suggest_categorical('layers', [[3,32,32,4], [3,128,128,4], [3,512,512, 4], [3,32,32,32,4], [3,128,128,128,4], [3,512,512,512,4]])
    elif cfg.model.model == "FCNN":
        k = 1
        layers =  trial.suggest_categorical('layers', [[3,32,32,4], [3,128,128,4], [3,512,512, 4], [3,32,32,32,4], [3,128,128,128,4], [3,512,512,512,4]])
    batch_norm =  trial.suggest_categorical('batch_norm', [True, False])

    # Cargar configuración
    outdir = Path(cfg.outdir) / cfg.data.red / cfg.model.model  /  datetime.now().isoformat().split('.')[0][5:].replace('T', '_')
    weights_dir = outdir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Guardar configuración
    cfg.model.layers = layers
    cfg.model.K = [k] * (len(layers)-1)
    cfg.data.normalize_X = norm_X
    cfg.training.dual_coefs = [c_1, c_2, c_3]
    cfg.training.lr = lr
    cfg.training.batch_size = batch_size
    cfg.training.batch_norm = batch_norm
    cfg.training.weight_decay = weight_dec
    cfg.model.dropout = drop
    OmegaConf.save(cfg, outdir / 'config.yaml')

    # Inicializar tensorboard
    writer = SummaryWriter(outdir)

    # Establecer dispositivo
    torch.manual_seed(cfg.training.seed)
    device = cfg.training.device

    # Cargar red
    import numpy as np
    edge_index, edge_weights, net = load_net(cfg.data.red, cfg.data.red_path, device)
    if cfg.data.red == "30":
        net.line["max_loading_percent"] *= 1.1
        net.ext_grid["min_q_mvar"] = -50
    else:
        net.trafo.tap_pos = 0.0
        net.bus.max_vm_pu = 1.1
        net.bus.min_vm_pu = 0.9
        net.line.max_i_ka /= 20
    pp.runpp(net)
    Y_bus = get_Ybus(net, device)
    Y_line = get_Yline(net, device)
    line = get_line(net, device)
    line_to_cpu = line[0].detach().cpu()
    max_ika = torch.Tensor(net.line.max_i_ka.values / 100 * np.sqrt(3) * net.bus.vn_kv.values[line_to_cpu[:len(net.line)]]).to(device)

    # Cargar datos
    train_loader, val_loader, test_loader, norm_x = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, cfg.data.red, device)
    print(len(val_loader))

    # Inicializar modelo
    torch.autograd.set_detect_anomaly(True)
    dual_variables = init_lamdas(net, cfg.training.dual_coefs, device)
    min_vector, max_vector = get_max_min_values(net, device)
    num_layers = len(cfg.model.layers) - 1
    num_nodes = len(net.bus)
    if cfg.model.model=="GNN":
        model = GNNUnsupervised(cfg.model.layers, edge_index, edge_weights, num_layers, cfg.model.K, min_vector, max_vector, num_nodes,cfg.model.dropout, batch_norm=cfg.training.batch_norm, use_edge_weights=cfg.training.use_edge_weights).to(device)
    elif cfg.model.model=="FCNN":
        cfg.model.K = -1
        model = FCNNUnsupervised(cfg.model.layers, edge_index, Y_bus, num_layers, cfg.model.K, min_vector, max_vector, num_nodes, cfg.model.dropout,batch_norm=cfg.training.batch_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, betas=cfg.training.betas, weight_decay=cfg.training.weight_decay)
    criterion = my_loss

    # Entrenar el modelo
    num_epochs = cfg.training.num_epochs
    best_loss = 1e20
    best_epoch = 0


    # Corrida por epocas
    for epoch in range(num_epochs):
        
        # Pasada train y val
        train_loss = run_epoch(model, train_loader, optimizer, criterion, Y_line, line, Y_bus, max_ika, dual_variables, norm_x, epoch, writer)
        val_loss = evaluate(model, val_loader, criterion, Y_line, line, Y_bus, max_ika, dual_variables, norm_x, epoch, writer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), weights_dir / 'best_model.pt')
            best_epoch = epoch

        # Early stopping
        if epoch - best_epoch > cfg.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Métricas de factibilidad y punto de ajuste de voltaje
    feas_metric, voltaje_set_metric, no_conv_count, gaps_mean, gaps_max = feas_and_volt_metric(best_model, val_loader, net, norm_x)
    
    data = {
        'model_name': str(outdir),
        'val_loss': val_loss,
        'feasibility_metric': feas_metric,
        'voltaje_setpoint_metric': voltaje_set_metric,
        'no_conv_count': no_conv_count, 
        'gaps_mean_val': gaps_mean,
        'gaps_max_val': gaps_max
    }
    with open(outdir / 'best_model_info.json', 'w') as file:
        json.dump(data, file)

    writer.add_hparams(
        {
            'lr': cfg.training.lr,
            'beta0': cfg.training.betas[0],
            'beta1': cfg.training.betas[1],
            'weight_decay': cfg.training.weight_decay,
            'optimizer': 'Adam',
            'batch_size': cfg.training.batch_size,
            'c_1': c_1,
            'c_2': c_2
        },
        {'hparam/val_loss': best_loss})
    writer.close()

    return voltaje_set_metric


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo')
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=256)

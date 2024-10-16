import torch
import torch.nn as nn
import json
import argparse
import optuna

from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.arquitecturas import GNN_global, FCNN_global, GNN_Local
from src.Data_loader import load_net, load_data
from src.train_eval import run_epoch, evaluate

import warnings
warnings.filterwarnings('ignore')


# DEFINIR OBJETIVO DE OPTUNA

def objective(trial):
    
    cfg = OmegaConf.load(args.cfg)
    
    # BUSQUEDA DE HIPERPARAMETROS
    
    ## Entrenamiento
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32,64,128])
    
    # Regularizacion
    weight_dec = trial.suggest_loguniform('weight_decay', 0.000001, 0.01)
    drop = trial.suggest_uniform('dropout', 0, 0.5)
    
    # Modelo
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    if cfg.model.model=='GNN_local':
        k = trial.suggest_int('k', 2, 8)
        layers =  trial.suggest_categorical('layers', [[4,16,16,2],[4,32,32,2], [4,64,64,2], [4,16,16,16,2],[4,32,32,32,2],[4,64,64,64,2], [4,16,16,16,16,2],[4,32,32,32,32,2],[4,64,64,64,64,2] ])
        cfg.model.K = [k] * (len(layers)-1)
    else:
        layers =  trial.suggest_categorical('layers', [[4,32,32,2], [4,128,128,2], [4,512,512, 2], [4,1024,1024,2], [4,32,32,32,2], [4,128,128,128,2], [4,512,512,512,2], [4,1024,1024,1024,2], [4,32,32,32,32,2], [4,128,128,128,128,2],[4,512,512,512,512,2], [4,1024,1024,1024,1024,2]])

    # Cargar cfg
    
    outdir = Path(cfg.outdir) / cfg.model.model /  datetime.now().isoformat().split('.')[0][5:].replace('T', '_')
    weights_dir = outdir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Guardar cfg
    cfg.model.layers = layers
    cfg.training.weight_decay = weight_dec
    cfg.model.dropout = drop
    cfg.model.batch_norm = batch_norm
    cfg.training.lr = lr
    cfg.training.batch_size = batch_size
    OmegaConf.save(cfg, outdir / 'config.yaml')

    # Tensorboard
    writer = SummaryWriter(outdir)

    # Settear device
    torch.manual_seed(cfg.training.seed)
    device = cfg.training.device

    # Levantar red electrica
    num_nodes, num_gens, edge_index, edge_weights, feature_mask,  min_max_values_v, min_max_values_q, net = load_net(cfg.data.red,cfg.data.red_path,device)
    
    # Levantar datos
    train_loader, val_loader, test_loader, norm_X, norm_y = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, cfg.data.normalize_Y,device)
    train_loader_no_norm, val_loader_no_norm, test_loader_no_norm, _, _ = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, False,device)

    # Definir modelo
    if cfg.model.model == 'FCNN_global':
        cfg.model.layers[0] *= num_nodes
        model = FCNN_global(cfg.model.layers,len(cfg.model.layers)-1,num_nodes, feature_mask, min_max_values_v, min_max_values_q, cfg.model.dropout, cfg.model.batch_norm, norm_X, norm_y).to(device)
    elif cfg.model.model == 'GNN_local':
        model = GNN_Local(cfg.model.layers,edge_index,edge_weights,len(cfg.model.layers)-1,cfg.model.K,feature_mask,num_nodes, min_max_values_v, min_max_values_q, cfg.model.dropout,cfg.model.batch_norm, norm_X, norm_y).to(device)

    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,betas=cfg.training.betas,weight_decay=cfg.training.weight_decay)
    criterion = nn.MSELoss()  # Change the loss function as needed
    
    # Entrenamiento
    best_acc = torch.inf
    best_epoch = 0

    # Iterar en epocas
    for epoch in range(cfg.training.num_epochs):
        
        # Train y val
        calculate_ploss_metric = False
        train_loss, train_metric_v, train_metric_sh = run_epoch(model, train_loader, optimizer, criterion,calculate_ploss_metric, net, epoch,writer)
        val_loss, val_metric_v, val_metric_sh, total_metric  = evaluate(model, val_loader, val_loader_no_norm, criterion, calculate_ploss_metric, net,norm_y, epoch,writer)
        print(f"Epoch {epoch+1}/{cfg.training.num_epochs}, Train Loss: {train_loss:.8f}, Train Metric V: {train_metric_v:.4f}, Train Metric sh: {train_metric_sh:.4f}, Val Loss: {val_loss:.8f}, Val Metric V: {val_metric_v:.4f}, Val Metric sh: {val_metric_sh:.4f}")

        # Si es mejor modelo guardar
        if val_loss < best_acc:
            best_acc = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), weights_dir / 'best_model.pt')
            data = {
            'model_name': str(outdir),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metric_V': val_metric_v,
            'val_metric_sh': val_metric_sh,
            'total_metric': total_metric
            }
            with open(outdir / 'best_model_info.json', 'w') as file:
                json.dump(data, file)           
            
        # Early stopping
        if epoch - best_epoch > cfg.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Guardar hiperaprametros
    writer.add_hparams(
        {
            'lr': cfg.training.lr,
            'beta0': cfg.training.betas[0],
            'beta1': cfg.training.betas[1],
            'weight_decay': cfg.training.weight_decay,
            'optimizer': 'Adam',
            'batch_size': cfg.training.batch_size,
        },
        {'hparam/accuracy': best_acc})
    writer.close()

    return best_acc


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo')
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=128)
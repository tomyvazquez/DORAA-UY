import torch
import torch.nn as nn
import optuna
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

# sys.path.append(str(Path(__file__).parents[1]))
from src.arquitecturas import FCNN_global, GNN_Local
from src.Data_loader import load_net, load_data
from src.metric import feas_and_volt_metric
from src.train_eval import run_epoch, evaluate

import json 

import warnings
warnings.filterwarnings('ignore')


def objective(trial):
    
    cfg = OmegaConf.load(args.cfg)
    
    # BUSQUEDA DE HIPERPARAMETROS
    
    ## Entrenamiento
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32,64,128])
    
    ## Regularizacion
    weight_dec = trial.suggest_loguniform('weight_decay', 0.000001, 0.01)
    drop = trial.suggest_uniform('dropout', 0, 0.5)
    
    ## Modelo
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    if cfg.model.model == "GNN_local":
        k = trial.suggest_int('k', 2, 7)
        layers =  trial.suggest_categorical('layers', [[3,32,32,1], [3,128,128,1], [3,512,512, 1], [3,32,32,32,1], [3,128,128,128,1], [3,512,512,512,1]])
    else:
        layers =  trial.suggest_categorical('layers', [[3,32,32,1], [3,128,128,1], [3,512,512, 1], [3,32,32,32,1], [3,128,128,128,1], [3,512,512,512,1], [3,32,32,32,32,1], [3,128,128,128,128,1], [3,512,512,512,512,1]])

    # Cargar configuración
    
    outdir = Path(cfg.outdir) / cfg.data.red / cfg.model.model /  datetime.now().isoformat().split('.')[0][5:].replace('T', '_')
    weights_dir = outdir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Guardar configuración
    
    cfg.model.layers = layers
    cfg.model.batch_norm = batch_norm
    if cfg.model.model == "GNN_local":
        cfg.model.K = [k] * (len(layers)-1)
    cfg.training.lr = lr
    cfg.training.batch_size = batch_size
    cfg.training.weight_decay = weight_dec
    cfg.model.dropout = drop
    OmegaConf.save(cfg, outdir / 'config.yaml')

    # TensorBoard
    writer = SummaryWriter(outdir)

    # Device
    torch.manual_seed(cfg.training.seed)
    device = cfg.training.device

    # Red electrica
    num_nodes, num_gens, edge_index, edge_weights, feature_mask, net = load_net(cfg.data.red,device)
    
    # Levantar Red
    train_loader, val_loader, test_loader, norm_X, norm_y = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, cfg.data.normalize_Y,device)
    train_loader_no_norm_y, val_loader_no_norm_y, test_loader_no_norm_y, _, _ = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, False,device)


    # Modelo
    if cfg.model.model == 'FCNN_global':
        cfg.model.layers[0] *= num_nodes
        model = FCNN_global(cfg.model.layers,len(cfg.model.layers)-1,num_nodes, feature_mask, cfg.model.dropout, cfg.model.batch_norm, norm_X, norm_y).to(device)
    elif cfg.model.model == 'GNN_local':
        model = GNN_Local(cfg.model.layers,edge_index,edge_weights,len(cfg.model.layers)-1,cfg.model.K,feature_mask,num_nodes, cfg.model.dropout, cfg.model.batch_norm, norm_X, norm_y).to(device)



    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,betas=cfg.training.betas,weight_decay=cfg.training.weight_decay)
    criterion = nn.MSELoss()  # Change the loss function as needed

    # Entrenar
    best_acc = 1e10
    best_epoch = 0
    last_train_metric_ploss = -1.
    last_val_metric_ploss = -1.

    for epoch in range(cfg.training.num_epochs):

        # Pasada de train y val
        train_loss, train_metric, train_metric_ploss = run_epoch(model, train_loader, optimizer, criterion,False, net, epoch,writer)
        val_loss, val_metric, val_metric_ploss  = evaluate(model, val_loader, val_loader_no_norm_y, criterion, False, net, epoch,writer, norm_y=norm_y)
        if train_metric_ploss != None:
            last_train_metric_ploss = train_metric_ploss
        if val_metric_ploss != None:
            last_val_metric_ploss = val_metric_ploss
        print(f"Epoch {epoch+1}/{cfg.training.num_epochs}, Train Loss: {train_loss:.8f}, Train Metric: {train_metric:.4f}, Train Ploss: {last_train_metric_ploss:.4f},  Val Loss: {val_loss:.8f}, Val Metric: {val_metric:.4f},  Val Ploss: {last_val_metric_ploss:.4f}")

        # Guardar mejor modelo
        if val_metric < best_acc:
            best_loss = val_loss
            best_acc = val_metric
            best_train_loss = train_loss
            best_train_metric = train_metric
            best_epoch = epoch
            best_model = model
            torch.save(best_model.state_dict(), weights_dir / 'best_model.pt')

        # Early stopping
        if epoch - best_epoch > cfg.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Evaluacion
    feas_metric, ploss_metric, no_conv_count = feas_and_volt_metric(best_model, val_loader, net)

    data = {
    'model_name': str(outdir),
    'val_loss': best_loss,
    'val_metric': best_acc,
    'val_p_loss': ploss_metric,
    'val_feas_metric': feas_metric,
    'train_loss': best_train_loss,
    'train_metric': best_train_metric
    }

    # Guardar data mejor modelo
    with open(outdir / 'best_model_info.json', 'w') as file:
        json.dump(data, file)
        
    # Guardar Hiperaparametros
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
    study.optimize(objective, n_trials=256)
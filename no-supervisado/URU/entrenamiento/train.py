
# Imports
import time
import numpy as np
import optuna
import torch
import json
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
import pandapower as pp
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
    # Levantar CFG
    cfg = OmegaConf.load(args.cfg)

    # BUSQUEDA DE HIPERPARAMETROS
    
    # Penalizaciones
    c_1 = trial.suggest_loguniform('c_1', 0.1, 300)
    c_2 = trial.suggest_loguniform('c_2',0.1,300)
    c_3 =  trial.suggest_loguniform('c_3', 1e-6, 1e-2)
    
    # Entrenamiento
    lr = trial.suggest_uniform('lr', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [128,256])
    
    # Data
    normalize_X = trial.suggest_categorical('normalize_X', [True,False])
    batch_norm =  trial.suggest_categorical('batch_norm', [True, False])
    
    # Regularizacion
    drop = trial.suggest_uniform('dropout', 0, 0.5)
    weight_dec = trial.suggest_loguniform('weight_decay', 0.000001, 0.01)

    # Arquitectura
    if cfg.model.arq == "GNN":
        k = trial.suggest_int('k', 2, 8)
        layers =  trial.suggest_categorical('layers', [[4,32,32,5], [4,128,128,5], [4,512,512, 5], [4,32,32,32,5], [4,128,128,128,5], [4,512,512,512,5]])
    elif cfg.model.arq == "FCNN":
        k = 1
        layers =  trial.suggest_categorical('layers', [[4,512,512, 5], [4,1024,1024, 5], [4,2048,2048, 5],[4,512,512,512,5], [4,1024,1024,1024,5], [4,2048,2048,2048,5],[4,512,512,512,512,5], [4,1024,1024,1024,1024,5], [4,2048,2048,2048,2048,5]])
    
    
    # Guardar configuración
    outdir = Path(cfg.outdir) / datetime.now().isoformat().split('.')[0][5:].replace('T', '_')
    weights_dir = outdir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    cfg.model.layers = layers
    cfg.model.K = [k] * (len(layers)-1)
    cfg.data.normalize_X = normalize_X 
    cfg.training.dual_coefs = [c_1, c_2, c_3]
    cfg.training.lr = lr
    cfg.training.batch_size = batch_size
    cfg.training.batch_norm = batch_norm
    cfg.training.weight_decay = weight_dec
    cfg.model.dropout = drop
    OmegaConf.save(cfg, outdir / 'config.yaml')

    # Inicializar tensorboard
    writer = SummaryWriter(outdir)

    # Settear GPU
    torch.manual_seed(cfg.training.seed)
    device = cfg.training.device

    # Cargar red electrica
    edge_index, edge_weights, net = load_net(cfg.data.red, cfg.data.red_path, device)
    pp.runpp(net)
    Y_bus = get_Ybus(net, device)
    Y_line = get_Yline(net, device)
    line = get_line(net, device)
    line_to_cpu = line[0].detach().cpu()
    max_ika = torch.Tensor(net.line.max_i_ka.values / 100 * np.sqrt(3) * net.bus.vn_kv.values[line_to_cpu[:len(net.line)]]).to(device)

    # Cargar datos
    train_loader, val_loader, test_loader, norm_x = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, cfg.data.red, device)
    torch.autograd.set_detect_anomaly(True)
    dual_variables = init_lamdas(net, cfg.training.dual_coefs, device)
    min_vector, max_vector = get_max_min_values(net, device)

    # Levantar predictor
    num_layers = len(cfg.model.layers) - 1
    num_nodes = len(net.bus)
    if cfg.model.arq == "GNN":
        model = GNNUnsupervised(cfg.model.layers, edge_index, Y_bus, num_layers, cfg.model.K, min_vector, max_vector, num_nodes,cfg.model.dropout, batch_norm=cfg.training.batch_norm).to(device)
    elif cfg.model.arq == "FCNN":
        model = FCNNUnsupervised(cfg.model.layers, edge_index, Y_bus, num_layers, cfg.model.K, min_vector, max_vector, num_nodes,cfg.model.dropout, batch_norm=cfg.training.batch_norm).to(device)
        k = trial.suggest_int('k', 1,1)

    # Levantar optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, betas=cfg.training.betas, weight_decay=cfg.training.weight_decay)
    criterion = my_loss

    # Entrenar el modelo
    num_epochs = cfg.training.num_epochs
    best_loss = torch.inf
    best_epoch = 0
    start_total = time.time()
    
    # Para cada epoca
    for epoch in range(num_epochs):
        
        # Train y validación
        start_time = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, Y_line, line, Y_bus, max_ika, dual_variables, norm_x, epoch, writer)
        val_loss = evaluate(model, val_loader, criterion, Y_line, line, Y_bus, max_ika, dual_variables, norm_x, epoch, writer)
        end_time = time.time()
        
        # Printear salida
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {end_time - start_time:.2f}s")

        # Guardar modelo si es mejor
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_epoch = epoch
            torch.save(best_model.state_dict(), weights_dir / 'best_model.pt')

        # Early stopping
        if epoch - best_epoch > cfg.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Métricas
    unfeasibles_val, ploss_val, ac_flow_error_val, ploss_pred_val, gaps_mean_val, gaps_max_val = feas_and_volt_metric(best_model, val_loader, net, norm_x)
    unfeasibles_train, ploss_train, ac_flow_error_train, ploss_pred_train, gaps_mean_train, gaps_max_train = feas_and_volt_metric(best_model, train_loader, net, norm_x)
    end_total = time.time()
    sec = end_total - start_total
    print('Total time: ', sec // 3600, 'h', (sec % 3600) // 60, 'm', sec % 60, 's')
    
    
    # Guardar info
    data = {
        'model_name': str(outdir),
        'val_loss': best_loss,
        'unfeasibles_val': unfeasibles_val,
        'ploss_val': ploss_val,
        'ac_flow_error_val': ac_flow_error_val,
        'ploss_pred_val': ploss_pred_val,
        'unfeasibles_train': unfeasibles_train,
        'ploss_train': ploss_train,
        'ac_flow_error_train': ac_flow_error_train,
        'ploss_pred_train': ploss_pred_train,
        'time': end_total - start_total,
        'gaps_mean_val': gaps_mean_val,
        'gaps_max_val': gaps_max_val,
        'gaps_mean_train': gaps_mean_train,
        'gaps_max_train': gaps_max_train
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
    return ploss_val


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo')
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=256)

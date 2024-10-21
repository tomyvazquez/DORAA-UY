import torch
import numpy as np
import torch.nn as nn
import pandapower as pp
import tqdm

from src.utils import get_Ybus, get_Yline, init_lamdas, get_line

import julia
julia.install()

from julia.api import Julia
jl = Julia(compiled_modules=False)


class NormalizedError(nn.Module):
    def __init__(self):
        super(NormalizedError, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Calcula el error normalizado entre la predicción y el valor real.

        Parámetros:
        y_pred (Tensor): Predicciones del modelo.
        y_true (Tensor): Valores reales.

        Retorna:
        Tensor: Error normalizado.
        """
        # Asegurarse de que las predicciones y los valores reales están en la CPU
        y_pred = y_pred.cpu()
        y_true = y_true.cpu()

        # Calcular la norma de la diferencia entre la predicción y el valor real
        numerator = torch.norm(y_pred - y_true,dim=1)

        # Calcular la norma del valor real
        denominator = torch.norm(y_true,dim=1)

        # # Evitar la división por cero
        # if denominator == 0:
        #     return torch.tensor(0.0)

        # Calcular y retornar el error normalizado
        return torch.mean(torch.sqrt(numerator / denominator))


class PlossMetric(nn.Module):
    def __init__(self, net):
        super(PlossMetric, self).__init__()
        self.net = net


    def forward(self, X, y):
        X_np = X.clone().detach().cpu().numpy()
        y_np = y.clone().detach().cpu().numpy()
        loss = 0
        converged = 0
        for i in range(X.shape[0]):
            # se agrega para que ande en la red de uru, tb anda en ieee. Lo de arriba era lo de antes
            id_load = [j for j, num in enumerate(self.net.bus.reset_index()['index'].to_list()) if num in self.net.load.bus.to_list()]
            id_gen = [j for j, num in enumerate(self.net.bus.reset_index()['index'].to_list()) if num in self.net.gen.bus.to_list()]
            # print("id_gen",id_gen)
            # print("id_load",id_load)
            # print("X_np",X_np[i])
            # print("y_np",y_np[i,id_gen][:,0])
            # print("net.gen",self.net.gen)
            # print('load p',X_np[i,id_load,0].shape)
            # print('load q',X_np[i,id_load,1].shape)
            self.net.load.loc[:,'p_mw'] = X_np[i,id_load,0]
            self.net.load.loc[:,'q_mvar'] = X_np[i,id_load,1]
            self.net.gen.loc[:,'p_mw'] =  X_np[i,id_gen,2]
            # print('gen p',X_np[i,id_gen,2].shape)
            self.net.gen.loc[:,'vm_pu'] =  y_np[i,id_gen][:,0]
            
            # print('vm_pu',y_np[i,id_gen,0].shape)
            # print("net.gen After",self.net.gen)

            try:
                pp.runpp(self.net, numba=False)
                loss += self.net.res_line.pl_mw.sum()
                converged += 1
            except:
                pass
        if converged!=0:
            loss /= converged
        else:
            loss = -1
        return loss

def relative_feas_error(x,x_min,x_max):
    error = np.max((np.zeros_like(x),x-x_max),axis=0) - np.min((np.zeros_like(x),x-x_min),axis=0)
    return (error/(x_max-x_min)).sum()

def get_stats(net,tol=1e-3, percent_gap=0):
    # METRICA
    Y_line_ij = np.asarray(net._ppc["internal"]["Yf"].todense())
    Y_line_ji = np.asarray(net._ppc["internal"]["Yt"].todense())
    V_mag = net.res_bus.vm_pu
    delta = net.res_bus.va_degree
    V = V_mag * np.exp(1j * delta * 2*np.pi/360)
    V_lines_to = [V[x] for x in net.line.to_bus] + [V[x] for x in net.trafo.lv_bus]
    V_lines_from = [V[x] for x in net.line.from_bus] + [V[x] for x in net.trafo.hv_bus]
    ploss = (V_lines_from * np.conj(np.matmul(Y_line_ij,V)) + V_lines_to * np.conj(np.matmul(Y_line_ji,V))).real * net.sn_mva
    ploss_cost = ploss.sum()
    
    # v_cost = (np.abs(net.res_bus.vm_pu.values - 1)**2).sum()
    max_line = net.line.max_loading_percent.values.copy()
    if percent_gap is not None:
        max_line = max_line * (1 + percent_gap)
    unfeas_line = (net.res_line.loading_percent.values > max_line + tol).sum()
    gap_line_max = (net.res_line.loading_percent.values - net.line.max_loading_percent.values - tol)/(net.line.max_loading_percent.values - 0)
    gap_line_max = gap_line_max[np.where(gap_line_max > 0)] * 100
    # if unfeas_line > 0:
        # print(gap_line_max)
    #     print((net.res_line.loading_percent.values - net.line.max_loading_percent.values)[np.where(net.res_line.loading_percent.values > net.line.max_loading_percent.values)], net.line.max_loading_percent.values[np.where(net.res_line.loading_percent.values > net.line.max_loading_percent.values)])
    
    unfeas_trafo = 0
    gap_trafo_max = []
    if len(net.trafo) > 0:
        max_trafo = net.trafo.max_loading_percent.values.copy()
        if percent_gap is not None:
            max_trafo = max_trafo * (1 + percent_gap)
        unfeas_trafo = (net.res_trafo.loading_percent.values > max_trafo + tol).sum()
        gap_trafo_max = (net.res_trafo.loading_percent.values - net.trafo.max_loading_percent.values - tol)/(net.trafo.max_loading_percent.values - 0)
        gap_trafo_max = gap_trafo_max[np.where(gap_trafo_max > 0)] * 100
    else:
        unfeas_trafo = 0
    
    max_volt = net.bus.max_vm_pu.values.copy()
    min_volt = net.bus.min_vm_pu.values.copy()
    if percent_gap is not None:
        max_volt +=  percent_gap*(net.bus.max_vm_pu.values - net.bus.min_vm_pu.values)
        min_volt -= percent_gap*(net.bus.max_vm_pu.values - net.bus.min_vm_pu.values)
        # print(f"Max Volt: {max_volt}")
        # print(f"Min Volt: {min_volt}")
    unfeas_volt = (net.res_bus.vm_pu.values < min_volt - tol).sum() + (net.res_bus.vm_pu.values > max_volt + tol).sum()
    # if unfeas_volt > 0:
        # print([(x,y) for x,y in zip(net.res_bus.vm_pu.values, net.bus.max_vm_pu.values)])
        # print(net.res_bus.vm_pu.values[np.where(net.res_bus.vm_pu.values > max_volt + tol)])
    gap_volt_max = (net.res_bus.vm_pu.values - net.bus.max_vm_pu.values - tol)/(net.bus.max_vm_pu.values - net.bus.min_vm_pu.values)
    gap_volt_max = gap_volt_max[np.where(gap_volt_max > 0)] * 100
    gap_volt_min = (-net.res_bus.vm_pu.values + net.bus.min_vm_pu.values + tol)/(net.bus.max_vm_pu.values - net.bus.min_vm_pu.values)
    gap_volt_min = gap_volt_min[np.where(gap_volt_min > 0)] * 100
    # print('gap_volt',gap_volt_max)
    
    max_q_extgrid = net.ext_grid.max_q_mvar.values.astype(float).copy()
    min_q_extgrid = net.ext_grid.min_q_mvar.values.astype(float).copy()
    if percent_gap is not None:
        max_q_extgrid +=  percent_gap*(net.ext_grid.max_q_mvar.values - net.ext_grid.min_q_mvar.values)
        min_q_extgrid -= percent_gap*(net.ext_grid.max_q_mvar.values - net.ext_grid.min_q_mvar.values)
    unfeas_q_ext_grid = (net.res_ext_grid.q_mvar.values < min_q_extgrid - tol).sum() + (net.res_ext_grid.q_mvar.values > max_q_extgrid + tol).sum()
    gap_q_ext_grid_max = (net.res_ext_grid.q_mvar.values - net.ext_grid.max_q_mvar.values - tol)/(net.ext_grid.max_q_mvar.values - net.ext_grid.min_q_mvar.values)
    gap_q_ext_grid_max = gap_q_ext_grid_max[np.where(gap_q_ext_grid_max > 0)] * 100
    gap_q_ext_grid_min = (-net.res_ext_grid.q_mvar.values + net.ext_grid.min_q_mvar.values + tol)/(net.ext_grid.max_q_mvar.values - net.ext_grid.min_q_mvar.values)
    gap_q_ext_grid_min = gap_q_ext_grid_min[np.where(gap_q_ext_grid_min > 0)] * 100

    # append all gaps if there is a value
    gaps_percentages = []
    if len(gap_line_max) > 0:
        gaps_percentages += list(gap_line_max)
    if len(gap_trafo_max) > 0:
        gaps_percentages += list(gap_trafo_max)
    if len(gap_volt_max) > 0:
        gaps_percentages += list(gap_volt_max)
    if len(gap_volt_min) > 0:
        gaps_percentages += list(gap_volt_min)
    if len(gap_q_ext_grid_max) > 0:
        gaps_percentages += list(gap_q_ext_grid_max)
    if len(gap_q_ext_grid_min) > 0:
        gaps_percentages += list(gap_q_ext_grid_min)
        
    if unfeas_line > 0 or unfeas_trafo > 0 or unfeas_volt > 0 or unfeas_q_ext_grid > 0:
        unfeas = True
    else:
        unfeas = False
    return ploss_cost, unfeas, gaps_percentages




def feas_and_volt_metric(model,val_loader,net, norm_x=None):

    idxs_gen = net.bus.index.get_indexer(list(net.gen.bus.values))
    idxs_load = net.bus.index.get_indexer(list(net.load.bus.values))
    idxs_shunts = net.bus.index.get_indexer(list(net.sgen[net.sgen['controllable']==True].bus.values))
    idxs_sgen = net.bus.index.get_indexer(list(net.sgen[net.sgen['controllable']==False].bus.values))

    unfeasible = 0
    ploss_metric_real = 0
    ploss_metric_pred = 0
    feas_count = 0
    ac_flow_error = 0
    gap_percentages = []
    for x in val_loader:
        output = model(x[0]).detach().cpu()
        x = x[0]
        p_ext_grid, q_gen, q_shunts, vm_pu_gen, ang_gen = output[:,:,0], output[:,:,1], output[:,:,2], output[:,:,3], output[:,:,4]
        batch_size = vm_pu_gen.shape[0]
        if norm_x is not None:
            mean = norm_x['mean']
            std = norm_x['std']
            std = torch.where(torch.isinf(std), torch.tensor(0.0), std)
            x = x * std + mean
        x = x.detach().cpu()
        p_load, q_load, p_gen, p_sgen = x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]
        for i in range(batch_size):
            try:
                net.gen.vm_pu = vm_pu_gen[i][idxs_gen].detach().cpu().numpy()
                net.sgen.loc[net.sgen['controllable']==True,"q_mvar"] = q_shunts[i][idxs_shunts].detach().cpu().numpy() * 100
                net.load.p_mw = p_load[i][idxs_load].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
                net.load.q_mvar = q_load[i][idxs_load].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
                net.gen.p_mw =p_gen[i][idxs_gen].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
                net.sgen.loc[net.sgen['controllable']==False, "p_mw"] = p_sgen[i][idxs_sgen].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
                pp.runpp(net,numba=False,enforce_q_lims=True)
                ploss_metric_real_i, unfeasible_i, gap_percentage =get_stats(net)
                ploss_metric_real += ploss_metric_real_i
                unfeasible += unfeasible_i
                feas_count += 1
                gap_percentages += gap_percentage
            except:
                pass
    if len(gap_percentages)==0:
        avg_gap = 0
        max_gap = 0
    else:
        avg_gap = np.mean(gap_percentages)
        max_gap = np.max(gap_percentages)
    return unfeasible/feas_count, ploss_metric_real/feas_count, ac_flow_error / len(val_loader), ploss_metric_pred/ len(val_loader), avg_gap, max_gap


# def feas_and_volt_metric(model,val_loader,pf_net,net):

#     idxs_gen = net.bus.index.get_indexer(list(net.gen.bus.values))
#     idxs_load = net.bus.index.get_indexer(list(net.load.bus.values))
#     idxs_shunts = net.bus.index.get_indexer(list(net.sgen[net.sgen['controllable']==True].bus.values))
#     idxs_sgen = net.bus.index.get_indexer(list(net.sgen[net.sgen['controllable']==False].bus.values))

#     unfeasible = 0
#     ploss_metric_real = 0
#     ploss_metric_pred = 0
#     feas_count = 0
#     ac_flow_error = 0
#     for x in val_loader:
#         output = model(x[0])
    
#         if norm_x is not None:
#             mean = norm_x['mean']
#             std = norm_x['std']
#             std = torch.where(torch.isinf(std), torch.tensor(0.0), std)
#             x[0] = x[0] * std + mean
#         Input = torch.cat((x[0],output),dim=-1)
#         # implicitas = pf_net(Input,desnormalizar=True,normalizar_in=True)

#         output = output.detach().cpu()
#         x = x[0].detach().cpu()
#         implicitas = implicitas.detach().cpu()
#         p_load, q_load, p_gen, p_sgen = x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]
#         V_mag_gen, q_shunt = output[:,:,0], output[:,:,1]
#         q_gen, V_mag_pq, delta, p_ext_grid = implicitas[:,:,0], implicitas[:,:,1], implicitas[:,:,2], implicitas[:,:,3]
        
#         # Trahseada
#         V_mag_ext_grid = np.zeros(V_mag_gen.shape)
#         V_mag_ext_grid[:,3] = 1.0

#         V_mag = V_mag_gen + V_mag_pq + V_mag_ext_grid

#         p = p_gen + p_ext_grid + p_sgen - p_load  
#         q = q_gen + q_shunt - q_load
#         V = V_mag * np.exp(1j * delta)
#         S = (p + 1j * q) / 100
        
        
#         # V = vm_pu_gen * (torch.cos(ang_gen) + torch.sin(ang_gen)*1j)
#         # S =  p_gen + p_ext_grid + p_sgen - p_load   + 1j * (q_gen + q_shunts - q_load)
#         # Y_bus = get_Ybus(net, V.device)
#         # ac_flow_error += abs((S - V * np.conj(np.dot(V, Y_bus.T)))).mean()
#         # line_to,line_from = get_line(net, V.device)
#         # Y_line_ij, Y_line_ji = get_Yline(net, V.device)
#         # V_to = V[:,line_to]
#         # V_from = V[:,line_from]
#         # ploss_metric_pred += ((V_from * np.conj(np.dot(V,Y_line_ij.t())) + V_to * torch.conj(np.dot(V,Y_line_ji.t()))).real).sum(dim=1).mean().item()
#         batch_size = V_mag_gen.shape[0]
#         for i in range(batch_size):
#             try:
#                 net.gen.vm_pu = V_mag_gen[i][idxs_gen].detach().cpu().numpy()
#                 net.sgen.loc[net.sgen['controllable']==True,"q_mvar"] = q_shunt[i][idxs_shunts].detach().cpu().numpy() * 100
#                 net.load.p_mw = p_load[i][idxs_load].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
#                 net.load.q_mvar = q_load[i][idxs_load].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
#                 net.gen.p_mw =p_gen[i][idxs_gen].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
#                 net.sgen.loc[net.sgen['controllable']==False, "p_mw"] = p_sgen[i][idxs_sgen].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
#                 pp.runpp(net,numba=False,enforce_q_lims=True)
#                 ploss_metric_real_i, unfeasible_i =get_stats(net)
#                 ploss_metric_real += ploss_metric_real_i
#                 unfeasible += unfeasible_i
#                 feas_count += 1
#             except:
#                 pass

#     return unfeasible/feas_count, ploss_metric_real/feas_count, ac_flow_error / len(val_loader), ploss_metric_pred/ len(val_loader)


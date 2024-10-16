import torch
import torch.nn as nn
import pandapower as pp
import numpy as np

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
        return torch.mean(numerator / denominator)


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
            self.net.load.loc[:,'p_mw'] = X_np[i,self.net.load["bus"].to_list(),0]
            self.net.load.loc[:,'q_mvar'] = X_np[i,self.net.load["bus"].to_list(),1]
            self.net.gen.loc[:,'p_mw'] =  X_np[i,self.net.gen["bus"].to_list(),2]
            self.net.gen.loc[:,'vm_pu'] =  y_np[i,self.net.gen["bus"].to_list(),0]
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


def get_stats(net,tol=1e-4):
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
    unfeas_line = (net.res_line.loading_percent.values > net.line.max_loading_percent.values + tol).sum()
    unfeas_trafo = 0
    if len(net.trafo) > 0:
        unfeas_trafo = (net.res_trafo.loading_percent.values > net.trafo.max_loading_percent.values + tol).sum()
    else:
        unfeas_trafo = 0
    unfeas_volt = (net.res_bus.vm_pu.values < net.bus.min_vm_pu.values - tol).sum() + (net.res_bus.vm_pu.values > net.bus.max_vm_pu.values + tol).sum()
    unfeas_q_ext_grid = (net.res_ext_grid.q_mvar.values < net.ext_grid.min_q_mvar.values - tol).sum() + (net.res_ext_grid.q_mvar.values > net.ext_grid.max_q_mvar.values + tol).sum()

    if unfeas_line > 0 or unfeas_trafo > 0 or unfeas_volt > 0 or unfeas_q_ext_grid > 0:
        unfeas = True
    else:
        unfeas = False
    return ploss_cost, unfeas


def feas_and_volt_metric(model,val_loader,net):

    idxs_gen = net.bus.index.get_indexer(list(net.gen.bus.values))
    idxs_load = net.bus.index.get_indexer(list(net.load.bus.values))


    feasibilty_metric = 0
    p_loss_metric = 0
    no_conv_count = 0
    feas_count = 0
    
    for x in val_loader:
        output = model(x[0], denormalize_y=True, training=False).detach().cpu()
        vm_pu_gen = output.squeeze()
        batch_size = vm_pu_gen.shape[0]

        for i in range(batch_size):
            net.gen.vm_pu = vm_pu_gen[i][idxs_gen].detach().cpu().numpy()
            net.load.p_mw = x[0][i,idxs_load,0].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
            net.load.q_mvar = x[0][i,idxs_load,1].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
            net.gen.p_mw = x[0][i,idxs_gen,2].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
            try:
                pp.runpp(net,numba=False, enforce_q_lims=True)
                p_loss_metric_i, feasibilty_metric_i = get_stats(net)
                p_loss_metric += p_loss_metric_i
                feasibilty_metric += feasibilty_metric_i
                feas_count += 1
            except:
                pass
    return feasibilty_metric/feas_count, p_loss_metric/feas_count, no_conv_count


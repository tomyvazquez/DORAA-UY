import torch
import torch.nn as nn
import pandapower as pp

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
        # # Asegurarse de que las predicciones y los valores reales están en la CPU
        # y_pred = y_pred.cpu()
        # y_true = y_true.cpu()

        # Calcular la norma de la diferencia entre la predicción y el valor real
        y_pred_sh = y_pred[:,:,1]
        y_true_sh = y_true[:,:,1]
        y_pred_v = y_pred[:,:,0]
        y_true_v = y_true[:,:,0]

        numerator_sh = torch.norm(y_pred_sh - y_true_sh,dim=1)
        numerator_v = torch.norm(y_pred_v - y_true_v,dim=1)
        denominator_sh = torch.norm(y_true_sh,dim=1)
        denominator_v = torch.norm(y_true_v,dim=1)

        e_v = torch.mean(numerator_v / denominator_v)
        e_sh = torch.mean(numerator_sh / denominator_sh)
        return e_v , e_sh

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



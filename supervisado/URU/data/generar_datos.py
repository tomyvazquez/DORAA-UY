import pandapower as pp
import pandapower.networks as nw
from copy import deepcopy
import pandapower.plotting as plot
import numpy as np
import pandas as pd
import os
import numba

import julia
julia.install()

from julia.api import Julia
jl = Julia(compiled_modules=False)

import warnings
warnings.filterwarnings('ignore')


# Levantar red
net = pp.from_pickle('red_uru.p')
net.line["pm_param/target_branch"] = True

# Levantar datos crudos
gen_p = pd.read_csv("./crudo/GENP_total_pandapower.csv")
load_p = pd.read_csv("./crudo/LOADP_total_pandapower.csv")

# Función que mapea el numero de bus a indice
bus_map = {net.bus.index[i]: i for i in range(len(net.bus))}
def bus_pos(buses):
    try:
        return [bus_map[bus] for bus in buses]
    except:
        return bus_map[buses]

# Empezar a generar datos
X = []
Y_volt = []
Y_react = []
Y_switch_shunts = []

for index in range(gen_p.shape[0]):
  # Inicializar en cero salidas
  y_volt = np.zeros((len(net.bus["name"]),1))
  y_react = np.zeros((len(net.bus["name"]),1))
  y_switch_shunts  = np.zeros((len(net.bus["name"]),1))
  X_i = np.zeros((len(net.bus["name"]),4))
  
  # Printear itearacion
  print('iteracion:', index)

  # Obtener indices de generadores, shunts y cargas
  gen_idx = net.gen.bus.values.astype(str)
  shunt_idx = net.sgen.loc[net.sgen.controllable==True].bus.values.astype(str)
  sgen_idx = net.sgen.loc[net.sgen.controllable==False].bus.values.astype(str)
  load_idx = net.load.bus.values.astype(str)

  # Cargar valores en la red
  net.gen.p_mw = gen_p.loc[index,gen_idx].values
  net.load.p_mw = load_p.loc[index,load_idx].values   
  hour = int(gen_p.loc[index,'Fecha'][-8:-6])
  if hour <= 6 and hour >= 1:
    coef_p = 0.995
  else:
    coef_p = 0.98
  net.load.q_mvar = load_p.loc[index,load_idx].values * np.tan(np.arccos(coef_p)) 
  net.sgen.p_mw.loc[net.sgen.controllable==False] = gen_p.loc[index,sgen_idx].values
  net.gen.max_p_mw = net.gen.p_mw 
  net.gen.min_p_mw = net.gen.p_mw

  # Rellenar entradas X
  X_i[bus_pos(net.gen.bus.values)] += np.array([np.zeros(len(net.gen.p_mw)), np.zeros(len(net.gen.p_mw)),net.gen.p_mw.values.astype('float64'), np.zeros(len(net.gen.p_mw))]).T
  X_i[bus_pos(net.load.bus.values)] += np.array([net.load.p_mw.values.astype('float64'), net.load.q_mvar.values.astype('float64'), np.zeros(len(net.load.q_mvar)), np.zeros(len(net.load.q_mvar))]).T
  X_i[bus_pos(net.sgen.bus.loc[net.sgen.controllable==False].values)] += np.array([np.zeros(len(net.sgen.loc[net.sgen.controllable==False].p_mw)), np.zeros(len(net.sgen.loc[net.sgen.controllable==False].p_mw)), np.zeros(len(net.sgen.loc[net.sgen.controllable==False].p_mw)),net.sgen.p_mw.loc[net.sgen.controllable==False].values.astype('float64')]).T

  # Resolver OPF
  try:
    pp.runpm_ploss(net)
    # Rellenar salidas Y
    y_volt[bus_pos(net.gen.bus.values)] = net.res_gen.vm_pu.values.reshape(-1,1)
    y_react[bus_pos(net.gen.bus.values)] = net.res_gen.q_mvar.values.reshape(-1,1)
    y_switch_shunts[bus_pos(net.sgen.bus.loc[net.sgen.controllable==True].values)] = net.res_sgen.loc[net.sgen.controllable==True].q_mvar.values.reshape(-1,1)
    Y_volt.append(y_volt)
    Y_react.append(y_react)
    Y_switch_shunts.append(y_switch_shunts)
    X.append(X_i)
  except:
    print("no convergio")

  # Guardar cada 1000
  if (index+1)%1000 == 0:
    X = np.array(X)
    Y_volt = np.array(Y_volt)
    Y_react = np.array(Y_react)
    Y_switch_shunts = np.array(Y_switch_shunts)
    np.save(f'./reduru/input_{(index+1)/1000}.npy',X)
    np.save(f'./reduru/vm_pu_opt_{(index+1)/1000}.npy',Y_volt)
    np.save(f'./reduru/q_mvar_opt_{(index+1)/1000}.npy',Y_react)
    np.save(f'./reduru/q_switch_shunt_opt_{(index+1)/1000}.npy',Y_switch_shunts)
    print("Guardado")
    X = []
    Y_volt = []
    Y_react = []
    Y_switch_shunts = []


# Guardar el total de datos
X = np.array(X)
Y_volt = np.array(Y_volt)
Y_react = np.array(Y_react)
Y_switch_shunts = np.array(Y_switch_shunts)

np.save(f'./reduru/input_{(index+1)//1000+1}.npy',X)
np.save(f'./reduru/vm_pu_opt_{(index+1)//1000+1}.npy',Y_volt)
np.save(f'./reduru/q_mvar_opt_{(index+1)//1000+1}.npy',Y_react)
np.save(f'./reduru/q_switch_shunt_opt_{(index+1)//1000+1}.npy',Y_switch_shunts)



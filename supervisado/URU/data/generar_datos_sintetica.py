import pandapower as pp
import pandapower.networks as nw
from copy import deepcopy
import pandapower.plotting as plot
import numpy as np
import pandas as pd
import os
import numba
import argparse
import julia
julia.install()

from julia.api import Julia
jl = Julia(compiled_modules=False)

import warnings
warnings.filterwarnings('ignore')


# Configurar argumentos de línea de comando
parser = argparse.ArgumentParser(description="Ejecutar simulación de pandapower con una red específica.")
parser.add_argument("--N", type=int, help="Especifica la cantidad de datos a generar")
args = parser.parse_args()

# Levantar red
net = pp.from_pickle('red_uru.p')
net.line["pm_param/target_branch"] = True

# Levantar datos crudos
gen_p = pd.read_csv("./crudo/GENP_total_pandapower.csv")
load_p = pd.read_csv("./crudo/LOADP_total_pandapower.csv")

# Calcular valores nominales como promedios
mean_gen =  gen_p.iloc[:,1:].mean()
mean_load =  load_p.iloc[:,1:].mean()

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

for index in range(args.N):
  
  # Inicializar en cero salidas
  y_volt = np.zeros((len(net.bus["name"]),1))
  y_react = np.zeros((len(net.bus["name"]),1))
  y_switch_shunts  = np.zeros((len(net.bus["name"]),1))
  X_i = np.zeros((len(net.bus["name"]),4))
  
  # Mostrar iteracion 
  print('iteracion:', index)
  
  # Obtener indices de los elementos
  gen_idx = net.gen.bus.values.astype(str)
  shunt_idx = net.sgen.loc[net.sgen.controllable==True].bus.values.astype(str)
  sgen_idx = net.sgen.loc[net.sgen.controllable==False].bus.values.astype(str)
  load_idx = net.load.bus.values.astype(str)
  
  # Obtener valores de entrada con uniforma centrado en valor nominal
  uniforme_activa_load = np.random.uniform(0.7,1.3,size=len(mean_load))
  uniforme_react_load = np.random.uniform(0.7,1.3,size=len(mean_load))
  uniforme_activa_gen = np.random.uniform(0.7,1.3,size=len(gen_idx))
  uniforme_activa_sgen = np.random.uniform(0.7,1.3,size=len(sgen_idx))

  # Cargar estos calores a la red
  net.gen.p_mw = mean_gen.loc[gen_idx].values*uniforme_activa_gen
  net.load.p_mw = mean_load.loc[load_idx].values*uniforme_activa_load
  coef_p = 0.98
  net.load.q_mvar = mean_load.loc[load_idx].values*uniforme_react_load * np.tan(np.arccos(coef_p))
  net.sgen.p_mw.loc[net.sgen.controllable==False] = mean_gen.loc[sgen_idx].values*uniforme_activa_sgen
  net.gen.max_p_mw = net.gen.p_mw 
  net.gen.min_p_mw = net.gen.p_mw

  # Rellenar entrada X
  X_i[bus_pos(net.gen.bus.values)] += np.array([np.zeros(len(net.gen.p_mw)), np.zeros(len(net.gen.p_mw)),net.gen.p_mw.values.astype('float64'), np.zeros(len(net.gen.p_mw))]).T
  X_i[bus_pos(net.load.bus.values)] += np.array([net.load.p_mw.values.astype('float64'), net.load.q_mvar.values.astype('float64'), np.zeros(len(net.load.q_mvar)), np.zeros(len(net.load.q_mvar))]).T
  X_i[bus_pos(net.sgen.bus.loc[net.sgen.controllable==False].values)] += np.array([np.zeros(len(net.sgen.loc[net.sgen.controllable==False].p_mw)), np.zeros(len(net.sgen.loc[net.sgen.controllable==False].p_mw)), np.zeros(len(net.sgen.loc[net.sgen.controllable==False].p_mw)),net.sgen.p_mw.loc[net.sgen.controllable==False].values.astype('float64')]).T
  # X_i[bus_pos(net.ext_grid.bus.values)] += np.array([0,0,0,0,net.ext_grid.vm_pu.values[0].astype('float64')]).reshape(1,5)

  # Resolver OPF
  try:
    pp.runpm_ploss(net)
    
    #Rellenar salidas y
    y_volt[bus_pos(net.gen.bus.values)] = net.res_gen.vm_pu.values.reshape(-1,1)
    y_react[bus_pos(net.gen.bus.values)] = net.res_gen.q_mvar.values.reshape(-1,1)
    y_switch_shunts[bus_pos(net.sgen.bus.loc[net.sgen.controllable==True].values)] = net.res_sgen.loc[net.sgen.controllable==True].q_mvar.values.reshape(-1,1)
    Y_volt.append(y_volt)
    Y_react.append(y_react)
    Y_switch_shunts.append(y_switch_shunts)
    X.append(X_i)
  except:
    print("no convergio")


# Transformar listas en arreglos
X = np.array(X)
Y_volt = np.array(Y_volt)
Y_switch_shunts = np.array(Y_switch_shunts)

# Concatenar arrays
output = np.concatenate((X, Y_volt, Y_switch_shunts), axis=2)

# Separar en train, val y test
np.random.shuffle(output)
train = output[0:int(output.shape[0]*0.77)]
val = output[int(output.shape[0]*0.77):int(output.shape[0]*0.93)]
test = output[int(output.shape[0]*0.93):]

# Creamos los directorios si no existen para train/val/test
if not os.path.exists('reduru_sintetica'+'/train'):
    os.makedirs('reduru_sintetica'+'/train')
if not os.path.exists('reduru_sintetica'+'/val'):
    os.makedirs('reduru_sintetica'+'/val')
if not os.path.exists('reduru_sintetica'+'/test'):
    os.makedirs('reduru_sintetica'+'/test')

# Guardamos los npy
# save train.npy
np.save('reduru_sintetica'+'/train/input.npy', train[:, :, :4])
np.save('reduru_sintetica'+'/train/vm_pu_opt.npy', train[:, :, 4:5])
np.save('reduru_sintetica'+'/train/q_switch_shunt_opt.npy', train[:, :, 5:6])
np.save('reduru_sintetica'+'/val/input.npy', val[:, :, :4])
np.save('reduru_sintetica'+'/val/vm_pu_opt.npy', val[:, :, 4:5])
np.save('reduru_sintetica'+'/val/q_switch_shunt_opt.npy', val[:, :, 5:6])
np.save('reduru_sintetica'+'/test/input.npy', test[:, :, :4])
np.save('reduru_sintetica'+'/test/vm_pu_opt.npy', test[:, :, 4:5])
np.save('reduru_sintetica'+'/test/q_switch_shunt_opt.npy', test[:, :, 5:6])
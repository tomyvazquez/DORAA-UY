import pandapower as pp
from copy import deepcopy
import numpy as np
import os

import julia
julia.install()
from julia.api import Julia
jl = Julia(compiled_modules=False)

import warnings
warnings.filterwarnings('ignore')

import argparse


# Configurar argumentos de línea de comando
parser = argparse.ArgumentParser(description="Ejecutar simulación de pandapower con una red específica.")
parser.add_argument("--red", type=str, choices=["30", "118"], help="Especificar el número de la red: '30' o '118'")
parser.add_argument("--N", type=int, help="Especifica la cantidad de datos a generar")
args = parser.parse_args()
red = args.red

# Fijar semilla
np.random.seed(0)

# Cambios a la red para corregir errores
if red == "30":
  net = pp.networks.case30()
  net.line["max_loading_percent"] *= 1.1
  net.ext_grid.min_q_mvar = -50

elif red == "118":
  net = pp.networks.case118()
  net.bus["max_vm_pu"] = 1.1
  net.bus["min_vm_pu"] = 0.9
  net.line["max_i_ka"] /= 20


# Permitir solo a los generadores cambiar su potencia al resolver opf
net.load['controllable'] = False
net.gen['controllable'] = True

# Definir valores nominales en la red
pp.runpm_ac_opf(net)
gen_p_nom = deepcopy(net.res_gen.p_mw.values)
load_p_nom = deepcopy(net.load['p_mw'])
load_q_nom = deepcopy(net.load['q_mvar'])

# Configurar lineas como target
net.line["pm_param/target_branch"] = True

# Empezar a generar datos
X = []
Y = []
X_LOAD = []


for i in range(args.N):
    y = np.zeros((len(net.bus["name"]),1))
    X_i = np.zeros((len(net.bus["name"]),3))
    print('iteracion:', i)
    
    
    # Tomar uniforme alrededor del valor nominal
    uniforme_activa_load = np.random.uniform(0.7,1.3,size=len(load_p_nom))
    uniforme_react_load = np.random.uniform(0.7,1.3,size=len(load_q_nom))
    uniforme_activa_gen = np.random.uniform(0.7,1.3,size=len(gen_p_nom))

    # Cargar los valores a la red
    net.load.loc[:,'p_mw'] = uniforme_activa_load*load_p_nom
    net.load.loc[:,'q_mvar'] = uniforme_react_load*load_q_nom
    net.gen.loc[:,'p_mw'] =  uniforme_activa_gen*gen_p_nom
    net.gen.loc[:,"max_p_mw"] = net.gen['p_mw']
    net.gen.loc[:,"min_p_mw"] = net.gen['p_mw']

    # Cargar los valores de entrada
    pp.runpp(net,numba=False)
    X_i[net.gen["bus"]] += np.array([np.zeros(len(net.res_gen.p_mw)), np.zeros(len(net.res_gen.p_mw)),net.res_gen.p_mw]).T
    X_i[net.load["bus"]] += np.array([net.res_load.p_mw, net.res_load.q_mvar, np.zeros(len(net.res_load.q_mvar,))]).T

    # Resolver OPF
    try:
      pp.runpm_ploss(net)
      
      # Cargar los valores de salida
      y[net.gen["bus"]] = net.res_gen.vm_pu.values.reshape(-1,1)
      Y.append(y)
      X.append(X_i)
    except:
      print("no convergio")


# Separar en train, test y validation

# Concatenar arrays
output = np.concatenate((X, Y), axis=2)

# Separar en train, val y test
np.random.shuffle(output)
train = output[0:int(output.shape[0]*0.77)]
val = output[int(output.shape[0]*0.77):int(output.shape[0]*0.93)]
test = output[int(output.shape[0]*0.93):]

# Creamos los directorios si no existen para train/val/test
if not os.path.exists('red'  + red +'/train'):
    os.makedirs('red' + red+'/train')
if not os.path.exists('red' + red +'/val'):
    os.makedirs('red' + red +'/val')
if not os.path.exists('red' + red +'/test'):
    os.makedirs('red' + red +'/test')

# Guardamos los npy
np.save('red' +red+'/train/input.npy', train[:, :, :3])
np.save('red' +red+'/train/vm_pu_opt.npy', train[:, :, 3])
np.save('red' +red+'/val/input.npy', val[:, :, :3])
np.save('red' +red+'/val/vm_pu_opt.npy', val[:, :, 3])
np.save('red' +red+'/test/input.npy', test[:, :, :3])
np.save('red' +red+'/test/vm_pu_opt.npy', test[:, :, 3])
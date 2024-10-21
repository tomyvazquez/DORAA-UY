
import torch
import numpy as np
# import pandapower as pp


# That also works for a list of buses
def bus_pos(buses, net):
  N = len(net.bus.index)
  bus_map = {net.bus.index[i]: i for i in range(N)}
  try:
      return [bus_map[bus] for bus in buses]
  except:
      return bus_map[buses]

def get_max_q(net):
  q_min = torch.zeros(len(net.bus)) 
  q_max = torch.zeros(len(net.bus))
  for idx in range(len(net.gen)):
    q_min[bus_pos(net.gen.iloc[idx].bus, net)] = net.gen.iloc[idx].min_q_mvar
    q_max[bus_pos(net.gen.iloc[idx].bus, net)] = net.gen.iloc[idx].max_q_mvar
  for idx in range(len(net.ext_grid)):
    q_min[bus_pos(net.ext_grid.iloc[idx].bus, net)] = net.ext_grid.iloc[idx].min_q_mvar
    q_max[bus_pos(net.ext_grid.iloc[idx].bus, net)] = net.ext_grid.iloc[idx].max_q_mvar
  return q_min/ net.sn_mva,q_max/net.sn_mva

# def get_max_min_values(net,device):
#     N = len(net.bus.index)

#     min_voltages_pu = torch.zeros(N)
#     max_voltages_pu = torch.zeros(N)

#     min_shunt_q = torch.zeros(N)
#     max_shunt_q = torch.zeros(N)

    
#     min_shunt_q[bus_pos(net.sgen.loc[net.sgen.controllable==True]["bus"].values.astype(int), net)] = torch.from_numpy(net.sgen.loc[net.sgen.controllable==True].min_q_mvar.values).float() /net.sn_mva
#     max_shunt_q[bus_pos(net.sgen.loc[net.sgen.controllable==True]["bus"].values.astype(int), net)] = torch.from_numpy(net.sgen.loc[net.sgen.controllable==True].max_q_mvar.values).float() /net.sn_mva


#     # Voltajes only for gens
#     min_voltages_pu[bus_pos(net.gen["bus"].values.astype(int), net)] = torch.from_numpy(net.bus.min_vm_pu.values).float()[bus_pos(net.gen["bus"].values.astype(int), net)]
#     max_voltages_pu[bus_pos(net.gen["bus"].values.astype(int), net)] = torch.from_numpy(net.bus.max_vm_pu.values).float()[bus_pos(net.gen["bus"].values.astype(int), net)]
    
#     min_vector = torch.vstack([min_voltages_pu, min_shunt_q]).t().to(device)
#     max_vector = torch.vstack([max_voltages_pu, max_shunt_q]).t().to(device)

#     return min_vector, max_vector


def get_max_min_values(net,device):
    N = len(net.bus.index)

    min_voltages_pu = torch.from_numpy(net.bus.min_vm_pu.values).float()
    max_voltages_pu = torch.from_numpy(net.bus.max_vm_pu.values).float()

    min_shunt_q = torch.zeros(N)
    max_shunt_q = torch.zeros(N)

    min_angles = -torch.ones(N) * 0.2 #np.pi / 2
    max_angles = torch.ones(N)* 0.2 #np.pi / 2

    min_p = -torch.zeros(N) 
    max_p = torch.zeros(N)

    min_q_gen,max_q_gen = get_max_q(net)
    
    min_shunt_q[bus_pos(net.sgen.loc[net.sgen.controllable==True]["bus"].values.astype(int), net)] = torch.from_numpy(net.sgen.loc[net.sgen.controllable==True].min_q_mvar.values).float() /net.sn_mva
    max_shunt_q[bus_pos(net.sgen.loc[net.sgen.controllable==True]["bus"].values.astype(int), net)] = torch.from_numpy(net.sgen.loc[net.sgen.controllable==True].max_q_mvar.values).float() /net.sn_mva


    # Fix for ext grid
    min_voltages_pu[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 1.0
    max_voltages_pu[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 1.0
    min_angles[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 0.0
    max_angles[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 0.0
    min_p[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = -10
    max_p[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 10

    min_vector = torch.vstack([min_p, min_q_gen, min_shunt_q, min_voltages_pu, min_angles]).t().to(device)
    max_vector = torch.vstack([max_p, max_q_gen, max_shunt_q, max_voltages_pu, max_angles]).t().to(device)

    return min_vector, max_vector

import torch

# def constraint_violation_power_flow(V_mag, V_ang, p, q, Y_bus):
#     # Separate real and imaginary parts to avoid using complex numbers directly
#     cos_ang = torch.cos(V_ang)
#     sin_ang = torch.sin(V_ang)
#     V_real = V_mag * cos_ang
#     V_imag = V_mag * sin_ang

#     # Compute S as real and imaginary parts
#     S_real = p
#     S_imag = q

#     # Use batch matrix multiplication efficiently
#     V = torch.stack((V_real, V_imag), dim=-1)  # Shape: (batch_size, n, 2)
    
#     # Repeat Y_bus for batch processing
#     Y_bus_repeated = Y_bus.repeat(V_mag.shape[0], 1, 1)
    
#     # Compute V * Y_bus
#     YV_real = torch.bmm(Y_bus_repeated.real, V_real.unsqueeze(2)) - torch.bmm(Y_bus_repeated.imag, V_imag.unsqueeze(2))
#     YV_imag = torch.bmm(Y_bus_repeated.real, V_imag.unsqueeze(2)) + torch.bmm(Y_bus_repeated.imag, V_real.unsqueeze(2))
    
#     # Compute AC equality constraint
#     AC_equality_real = S_real - (V_real.unsqueeze(2) * YV_real - V_imag.unsqueeze(2) * YV_imag).squeeze(2)
#     AC_equality_imag = S_imag - (V_real.unsqueeze(2) * YV_imag + V_imag.unsqueeze(2) * YV_real).squeeze(2)

#     return AC_equality_real, AC_equality_imag

# # def equality_penalty(U):
# #     return U**2

# def cost_function_voltage(V_mag):
#     # More efficient calculation for the voltage cost
#     cost = (V_mag - 1)**2
#     return torch.sum(cost, axis=-1)

# def my_loss(U, X, Y_line, Y_bus, ika_max, dual_variables):
#     dual_acflow_real = dual_variables[0]
#     dual_acflow_imag = dual_variables[1]
#     dual_lines = dual_variables[2]

#     # Split U into its components
#     U = U[0]
    
    
#     if norm_x is not None:
#       mean = norm_x['mean']
#       std = norm_x['std']
#       std = torch.where(torch.isinf(std), torch.tensor(0.0), std)
#       U = U * std + mean
    
#     p_load = U[:,:,0]
#     q_load = U[:,:,1]
#     p_gen = U[:,:,2]
#     p_sgen = U[:,:,3]

#     # Split X into its components
#     p_ext_grid = X[:,:,0]
#     q_gen = X[:,:,1]
#     q_shunt = X[:,:,2]
#     V_mag = X[:,:,3]
#     delta = X[:,:,4]

#     # Calculate power components
#     p = p_gen + p_ext_grid + p_sgen - p_load  
#     q = q_gen + q_shunt - q_load 

#     # Compute AC flow penalties
#     AC_flow_real, AC_flow_imag = constraint_violation_power_flow(V_mag, delta, p, q, Y_bus)

#     AC_flow_penalty_real = equality_penalty(AC_flow_real)
#     AC_flow_penalty_imag = equality_penalty(AC_flow_imag)

#     # Compute objective cost
#     objective = cost_function_voltage(V_mag)

#     # Sum of all penalties
#     loss = objective + torch.sum(AC_flow_penalty_real * dual_acflow_real, dim=-1) + torch.sum(AC_flow_penalty_imag * dual_acflow_imag, dim=-1)

#     # Compute mean loss
#     loss = torch.mean(loss)
#     return loss

def constraint_violation_power_flow(V_mag,V_ang,p,q,Y_bus):
  V = V_mag * (torch.cos(V_ang) + torch.sin(V_ang)*1j)
  S = p + 1j * q
  AC_equality = S - torch.bmm(torch.diag_embed(V), torch.conj(torch.bmm(Y_bus.repeat(V.shape[0],1,1),V.unsqueeze(2)))).squeeze()
  return AC_equality

def equality_penalty(U):
    Y = U**2
    return Y

def inequality_penalty(U, s=10):
    mask = U <= -1/s
    Y = torch.empty_like(U)
    Y[mask] = -torch.log(-U[mask])
    Y[~mask] = s * (U[~mask] + 1/s) - np.log(1/s)
    return Y

def cost_function_voltage(V_mag):
  ''' U is the output of the GNN, BxNx3 (Qgen, V, angle)
      a and b Nx3 are the upper and lower limits for the three magnitudes '''
  cost = (V_mag - 1)**2
  return torch.sum(cost,axis=-1)

def cost_ploss(V, V_to, V_from, Y_line_ij, Y_line_ji):
  ploss = (V_from * torch.conj(torch.mm(V,Y_line_ij.t())) + V_to * torch.conj(torch.mm(V,Y_line_ji.t()))).real# * 100 #net.sn_mva
  return ploss.sum(dim=1)
  

def my_loss(U,X,Y_line,Y_bus,Iij_max,dual_variables,line_to,line_from,norm_x):
  dual_acflow_real = dual_variables[0]
  dual_acflow_imag = dual_variables[1]
  dual_lines = dual_variables[2]

  Y_line_ij = Y_line[0]
  Y_line_ji = Y_line[1]
  
  U = U[0]
  
  if norm_x is not None:
      mean = norm_x['mean']
      std = norm_x['std']
      std = torch.where(torch.isinf(std), torch.tensor(0.0), std)
      U = U * std + mean
  p_load = U[:,:,0]
  q_load = U[:,:,1]
  p_gen = U[:,:,2]
  p_sgen = U[:,:,3]

  p_ext_grid = X[:,:,0]
  q_gen = X[:,:,1]
  q_shunt = X[:,:,2]
  V_mag = X[:,:,3]
  delta = X[:,:,4]
  
  p = p_gen + p_ext_grid + p_sgen - p_load  
  q = q_gen + q_shunt - q_load
  V = V_mag * torch.exp(1j * delta)
  V_to = V[:,line_to]
  V_from = V[:,line_from]
  
  # AC flow penalty
  AC_flow = constraint_violation_power_flow(V_mag, delta, p, q, Y_bus)
  AC_flow_penalty_real = equality_penalty(torch.real(AC_flow))
  AC_flow_penalty_imag = equality_penalty(torch.imag(AC_flow))



  # Inequality penalty
  lines_violation_ij = torch.abs(torch.conj(torch.mm(V,Y_line_ij.t())))[:,:len(Iij_max)] - Iij_max
  lines_violation_ji = torch.abs(torch.conj(torch.mm(V,Y_line_ji.t())))[:, :len(Iij_max)]- Iij_max

  lines_violation_penalty_ij = inequality_penalty(lines_violation_ij)
  lines_violation_penalty_ji = inequality_penalty(lines_violation_ji)

  # Objective cost
  objective = cost_ploss(V, V_to, V_from, Y_line_ij, Y_line_ji)

  # Sum of all penalties
  loss =  objective  + torch.mv(AC_flow_penalty_real,dual_acflow_real) +  torch.mv(AC_flow_penalty_imag,dual_acflow_imag) +  torch.mv(lines_violation_penalty_ij,dual_lines) + torch.mv(lines_violation_penalty_ji,dual_lines)  

  loss = torch.mean(loss)
  return loss





def my_loss_net_pf(U,X,pf_net, Y_line,dual_variables,line_to,line_from):
  dual_acflow_real = dual_variables[0]
  dual_acflow_imag = dual_variables[1]
  dual_lines = dual_variables[2]

  Y_line_ij = Y_line[0]
  Y_line_ji = Y_line[1]
  
  U = U[0]

  Input = torch.cat((U,X ),dim=-1)
  
  Z = pf_net(Input,desnormalizar=True,normalizar_in=True)


  p_load = U[:,:,0]
  q_load = U[:,:,1]
  p_gen = U[:,:,2]
  p_sgen = U[:,:,3]

  V_mag_gen = X[:,:,0]
  q_shunt = X[:,:,1]
  
  q_gen = Z[:,:,0]
  V_mag_pq = Z[:,:,1]
  delta = Z[:,:,2]
  p_ext_grid = Z[:,:,3]

  # Trahseada
  V_mag_ext_grid = torch.zeros(V_mag_gen.shape).to(V_mag_gen.device)
  V_mag_ext_grid[:,3] = 1.0
  
  
  V_mag = V_mag_gen + V_mag_pq + V_mag_ext_grid
  # print(f"V_mag_gen: {V_mag_gen[0]}")
  # print(f"V_mag_load: {V_mag_pq[0]}")
  # print(f"V_mag: {V_mag[0]}")
  # print(f"delta: {delta[0]}")
  p = p_gen + p_ext_grid + p_sgen - p_load  
  q = q_gen + q_shunt - q_load
  V = V_mag * torch.exp(1j * delta)
  
  
  # Objective cost

  V_to = V[:,line_to]
  V_from = V[:,line_from]
  objective = cost_ploss(V, V_to, V_from, Y_line_ij, Y_line_ji)


  # Sum of all penalties
  loss =  objective

  loss = torch.mean(loss)
  
  return loss
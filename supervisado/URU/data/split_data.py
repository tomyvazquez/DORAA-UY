import numpy as np
import os

# List of .npy files to merge
data_path = './reduru_final'

file_list_input = [f'input_{i}.0.npy' for i in range(1,24)]
file_list_q_shunt = [f'q_switch_shunt_opt_{i}.0.npy' for i in range(1,24)]
file_list_vm_pu = [f'vm_pu_opt_{i}.0.npy' for i in range(1,24)]

# Load the first file to initialize the merged array
input_list = []
q_shunt_list = []
vm_pu_list = []

# Iterate over the remaining files and concatenate them to the merged array
for idx,file in enumerate(file_list_input):
    input_list.append(np.load(os.path.join(data_path,file)))
    q_shunt_list.append(np.load(os.path.join(data_path,file_list_q_shunt[idx])))
    vm_pu_list.append(np.load(os.path.join(data_path,file_list_vm_pu[idx])))

merged_input = np.concatenate(input_list, axis=0)
merged_q_shunt = np.concatenate(q_shunt_list, axis=0)
merged_vm_pu = np.concatenate(vm_pu_list, axis=0)


# Save the merged array to a new .npy file
np.save(data_path+'input.npy', merged_input)
np.save(data_path+'q_switch_shunt_opt.npy', merged_q_shunt)
np.save(data_path+'vm_pu_opt.npy', merged_vm_pu)

output = np.concatenate((merged_input, merged_vm_pu, merged_q_shunt), axis=2)

# Do train, test and validation split
train = output[0:int(output.shape[0]*0.77)] # Hasta abril de 2022
val = output[int(output.shape[0]*0.77):int(output.shape[0]*0.95)]
test = output[int(output.shape[0]*0.95):]

# create train, val and test folders
os.makedirs(os.path.join(data_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'val'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'test'), exist_ok=True)

# save train.npy
np.save(data_path+'/train/input.npy', train[:, :, :4])
np.save(data_path+'/train/vm_pu_opt.npy', train[:, :, 4:5])
np.save(data_path+'/train/q_switch_shunt_opt.npy', train[:, :, 5:6])
np.save(data_path+'/val/input.npy', val[:, :, :4])
np.save(data_path+'/val/vm_pu_opt.npy', val[:, :, 4:5])
np.save(data_path+'/val/q_switch_shunt_opt.npy', val[:, :, 5:6])
np.save(data_path+'/test/input.npy', test[:, :, :4])
np.save(data_path+'/test/vm_pu_opt.npy', test[:, :, 4:5])
np.save(data_path+'/test/q_switch_shunt_opt.npy', test[:, :, 5:6])
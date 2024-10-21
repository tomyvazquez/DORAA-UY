from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import glob
import os
import argparse
import json

# path = "/home/iboero/Documents/GNN4OPF/Entrenar/runs/30/FCNN_global/"

def find_best_run_from_tfevents(path, k=10):
    '''
    Encuentra los mejores k entrenamientos en un directorio, a partir de los events.out.tfevents
    '''
    last = []
    entrenamiento = []

    for file in os.listdir(path):
        evento = glob.glob(path + file + '/' + "events.out.tfevents*")[0]
        event_loader = EventFileLoader(evento)
        events = event_loader.Load()
        for event in events:
            evento_final = event
        try:
            last.append(evento_final.summary.value[0].tensor.float_val[0])
            entrenamiento.append(file)
        except:
            pass

    indices_ordenados = sorted(range(len(last)), key=lambda i: last[i], reverse=False)
    best_values = indices_ordenados[:k]
    k_best_trains = [entrenamiento[i] for i in best_values]

    return k_best_trains



def find_best_run(path, k=10):
    '''
    Encuentra los mejores k entrenamientos en un directorio, a partir de los events.out.tfevents
    '''
    val_losses = []
    val_metrics = []
    val_p_losses = []
    entrenamiento = []

    for file_path in os.listdir(path):

        try:
            with open(os.path.join(path, file_path, 'best_model_info.json'), 'r') as file:
                data = json.load(file)
            
            entrenamiento.append(file_path)
            val_losses.append(data['val_loss'])
            val_metrics.append(data['val_metric'])
            val_p_loss.append(data['val_p_loss'])
        except:
            pass

    indices_ordenados = sorted(range(len(val_metrics)), key=lambda i: val_metrics[i], reverse=False)
    best_values = indices_ordenados[:k]
    k_best_trains = [entrenamiento[i] for i in best_values]
    k_best_train_values = [val_metrics[i] for i in best_values]
    


    if 'best' not in k_best_trains[0]:
        os.rename(os.path.join(path, k_best_trains[0]), os.path.join(path, 'best-'+k_best_trains[0]))

    return zip(k_best_trains, k_best_train_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encuentra los mejores k entrenamientos en un directorio, a partir de los events.out.tfevents')
    parser.add_argument('folder_path', type=str, help='Carpeta con todos los entrenamientos')
    parser.add_argument('-k', type=int, default=10, help='Número de los mejores entrenamientos a encontrar (por defecto: 10)')
    # parser.add_argument('metric', type=str, default='val_ploss', help='Carpeta con todos los entrenamientos')

    args = parser.parse_args()

    best = find_best_run(args.folder_path, args.k)
    #unzip best
    for b in best:
        print(b)

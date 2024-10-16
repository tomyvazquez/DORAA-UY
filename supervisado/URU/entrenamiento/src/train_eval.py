import torch
from src.metric import NormalizedError

def run_epoch(model, train_loader, optimizer, criterion, calculate_ploss_metric,net, epoch=None,writer=None):
    model.train()
    total_loss = 0
    total_metric_v = 0
    total_metric_sh = 0
    normalized_error = NormalizedError()
    if calculate_ploss_metric:
        indices=torch.randint(0,len(train_loader),(5,))
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data[0])
        target = data[1]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calcular la métrica

        metric  = normalized_error(output, target)
        total_metric_v += metric[0].item()
        total_metric_sh += metric[1].item()

    avg_loss = total_loss / len(train_loader)
    avg_metric_v = total_metric_v / len(train_loader)
    avg_metric_sh = total_metric_sh / len(train_loader)

    writer.add_scalar('Train/Training Loss', avg_loss, epoch)
    writer.add_scalar('Train/Training Metric V', avg_metric_v, epoch)
    writer.add_scalar('Train/Training Metric Sh', avg_metric_sh, epoch)

    return avg_loss, avg_metric_v, avg_metric_sh


# Definimos la funcion de evaluacion
def evaluate(model, data_loader, data_loader_no_norm_y, criterion,calculate_ploss_metric,net,norm_y,epoch=None,writer=None,test=False):
    model.eval()
    total_loss = 0
    total_metric_v = 0
    total_metric_sh = 0
    total_metric = 0
    normalized_error = NormalizedError()
    # Tomo 5 indice al azar entre 0 y el largo de data loader
    if calculate_ploss_metric and not test:
        indices=torch.randint(0,len(data_loader),(5,))
    elif calculate_ploss_metric and test:
        indices = range(len(data_loader))
    with torch.no_grad():
        for idx, data in enumerate(data_loader): 
            output = model(data[0])
            target = data[1]
            loss = criterion(output, target)  # target should contain true values for nodes with missing features
            total_loss += loss.item()
            #Calcular la métrica
            metric_v, metric_sh  = normalized_error(output, target)
            total_metric += metric_v.item()/2 + metric_sh.item()/2
    with torch.no_grad():
        for idx, data in enumerate(data_loader_no_norm_y):
            if norm_y is not None:
                output = model(data[0], training=False, denormalize_y=True)
            else:
                output = model(data[0], training=False)
            target = data[1]      
            # Calcular la métrica
            metric_v, metric_sh  = normalized_error(output, target)
            total_metric_v += metric_v.item()
            total_metric_sh += metric_sh.item()


    avg_loss = total_loss / len(data_loader)
    avg_metric_v = total_metric_v / len(data_loader)
    avg_metric_sh = total_metric_sh / len(data_loader)
    total_metric = total_metric / len(data_loader)


    if not test:
        if epoch != None:
            writer.add_scalar('Val/Validation Loss', avg_loss, epoch)
            writer.add_scalar('Val/Validation Metric V', avg_metric_v, epoch)
            writer.add_scalar('Val/Validation Metric Sh', avg_metric_sh, epoch)

    else:
        writer.add_scalar('Test/Test Loss', avg_loss, epoch)
        writer.add_scalar('Test/Test Metric V', avg_metric_v, epoch)
        writer.add_scalar('Test/Test Metric Sh', avg_metric_sh, epoch)

    return avg_loss, avg_metric_v, avg_metric_sh, total_metric

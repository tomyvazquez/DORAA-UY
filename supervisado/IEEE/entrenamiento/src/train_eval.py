import torch
from src.metric import NormalizedError, PlossMetric

def run_epoch(model, train_loader, optimizer, criterion,calculate_ploss_metric,net, epoch=None,writer=None):
    model.train()
    total_loss = 0
    total_metric = 0
    normalized_error = NormalizedError()
    ploss_metric = PlossMetric(net)
    total_metric_ploss = 0
    if calculate_ploss_metric:
        indices=torch.randint(0,len(train_loader),(5,))
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data[0])
        target = data[1].unsqueeze(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calcular la métrica
        metric = normalized_error(output, target)
        total_metric += metric.item()
        if calculate_ploss_metric and idx in indices:
            metric_ploss = ploss_metric(data[0], output)
            total_metric_ploss += metric_ploss
        else:
            avg_metric_ploss = None
    if calculate_ploss_metric:
        avg_metric_ploss = total_metric_ploss / len(indices)
        writer.add_scalar('Train/Training Ploss', avg_metric_ploss, epoch)
    avg_loss = total_loss / len(train_loader)
    avg_metric = total_metric / len(train_loader)

    writer.add_scalar('Train/Training Loss', avg_loss, epoch)
    writer.add_scalar('Train/Training Metric', avg_metric, epoch)

    return avg_loss, avg_metric, avg_metric_ploss


# Definimos la funcion de evaluacion
def evaluate(model, data_loader, data_loader_no_norm, criterion,calculate_ploss_metric,net,epoch=None,writer=None,test=False,norm_y=None): 
    model.eval()
    total_loss = 0
    total_metric = 0
    total_metric_ploss = 0
    normalized_error = NormalizedError()
    ploss_metric = PlossMetric(net)
    # Tomo 5 indice al azar entre 0 y el largo de data loader
    if calculate_ploss_metric and not test:
        indices=torch.randint(0,len(data_loader),(5,))
    elif calculate_ploss_metric and test:
        indices = range(len(data_loader))
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            output = model(data[0])
            target = data[1].unsqueeze(-1)
            loss = criterion(output, target)  # target should contain true values for nodes with missing features
            total_loss += loss.item()
    with torch.no_grad():
        for idx, data in enumerate(data_loader_no_norm):
            if norm_y != None:
                output = model(data[0], training=False, denormalize_y=True)
            else:
                output = model(data[0], training=False)
            target = data[1].unsqueeze(-1)        
            # Calcular la métrica
            metric = normalized_error(output, target)
            total_metric += metric.item()
            
            if calculate_ploss_metric and idx in indices:
                metric_ploss = ploss_metric(data[0], output)
                total_metric_ploss += metric_ploss
            else:
                avg_metric_ploss = None
    avg_loss = total_loss / len(data_loader)
    avg_metric = total_metric / len(data_loader)
    if calculate_ploss_metric:
        avg_metric_ploss = total_metric_ploss / len(indices)
    if not test:
        if epoch != None:
            writer.add_scalar('Val/Validation Loss', avg_loss, epoch)
            writer.add_scalar('Val/Validation Metric', avg_metric, epoch)
            if calculate_ploss_metric:
                writer.add_scalar('Val/Validation Ploss', avg_metric_ploss, epoch)
    else:
        writer.add_scalar('Test/Test Loss', avg_loss, epoch)
        writer.add_scalar('Test/Test Metric', avg_metric, epoch)
        writer.add_scalar('Test/Test Ploss', avg_metric_ploss, epoch)

    return avg_loss, avg_metric, avg_metric_ploss

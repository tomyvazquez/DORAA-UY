import torch
import time
import torch.autograd.profiler as profiler

def run_epoch(model, train_loader, optimizer, criterion, Y_line, line, Y_bus, Sij_max,dual_variables,norm_x,epoch,writer):
    model.train()
    total_loss = 0
    line_to, line_from = line

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data[0])
        # loss = criterion(data, output, Y_line, Y_bus, Sij_max,dual_variables, line_to, line_from)  # target should contain true values for nodes with missing features
        loss = criterion(data,output, Y_line,Y_bus,Sij_max,dual_variables,line_to,line_from,norm_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Train/Training Loss', avg_loss, epoch)

    return avg_loss

def evaluate(model, data_loader, criterion,Y_line, line, Y_bus, Sij_max, dual_variables,norm_x, epoch,writer,test=False):
    model.eval()
    total_loss = 0
    line_to, line_from = line

    with torch.no_grad():
        for data in data_loader:
            output = model(data[0])
            # loss = criterion(data, output, Y_line, Y_bus, Sij_max,dual_variables, line_to, line_from)  # target should contain true values for nodes with missing features
            loss = criterion(data,output,Y_line, Y_bus,Sij_max,dual_variables,line_to,line_from,norm_x)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    if test:
        writer.add_scalar('Test/Test Loss', avg_loss, epoch)
    else:
        writer.add_scalar('Val/Validation Loss', avg_loss, epoch)

    return avg_loss

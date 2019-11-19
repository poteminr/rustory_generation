import time
import math
import torch
import torch.nn.functional as F
import gc
from livelossplot import PlotLosses


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def callback(train_loss, val_loss, train_time, epoch_value, epoch_n):

    msg = 'Time: {} | Epoch: {} / {} | T-Loss: {:.3f} | Val-Loss: {:.3f}'.format(train_time, epoch_n, epoch_value,
                                                                         train_loss, val_loss)
    print(msg)


def cross_entropy(pred, target):

    pred_flat = pred.view(-1, pred.shape[-1])  
    target_flat = target.view(-1)  

    return F.cross_entropy(pred_flat, target_flat, ignore_index=0)

def reduce_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      patience=20,
                                                      factor=0.5,
                                                      verbose=True)

def train_loop(model, device, optimizer, train_loader, test_loader, lr_scheduler=reduce_lr_scheduler,
               criterion=cross_entropy, epoch_value=10, plot_loss=False):
    lr_policy = lr_scheduler(optimizer)
    start = time.time()
    liveloss = PlotLosses()

    for epoch_ind in range(epoch_value):
        try:
            logs = {}
            model.train()
            train_loss = 0

            for ind, (input_s, target_s) in enumerate(train_loader):
                input_s = input_s.to(device)
                target_s = target_s.to(device)

                pred = model(input_s)
                loss = criterion(pred, target_s)
                model.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss

                del input_s, target_s
                gc.collect()
                torch.cuda.empty_cache() 

            train_loss /= (ind + 1)


            test_loss = 0
            model.eval()

            with torch.no_grad():
                for ind, (input_s, target_s) in enumerate(test_loader):
                    input_s = input_s.to(device)
                    target_s = target_s.to(device)

                    pred = model(input_s)
                    loss = criterion(pred, target_s)

                    test_loss += loss

                    del input_s, target_s
                    gc.collect()
                    torch.cuda.empty_cache() 

                test_loss /= (ind + 1)
                lr_policy.step(test_loss)

            train_time = time_since(start)
            
            if plot_loss:
                logs['val_' + 'loss'] = train_loss
                logs['loss'] = test_loss
                liveloss.update(logs)
                liveloss.draw()
                
            else:
                callback(train_loss, test_loss, train_time, epoch_value, epoch_ind+1)
            
        except KeyboardInterrupt:
            print(f"Early stopping | Epoch: {epoch_ind + 1}")
            break
        
    return model
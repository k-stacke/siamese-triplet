import os
import numpy as np
import torch
from torch.cuda import amp
import time



def fit(args, train_loader, val_loader, model, loss_fn, optimizer, scheduler,
        scaler, n_epochs, device, log_interval, metrics=[],
        exp=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    # for epoch in range(0, start_epoch):
    #     scheduler.step()

    for epoch in range(args.start_epoch, n_epochs):
        start_time = time.time()
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn,
            optimizer, device, log_interval, metrics, scaler)
        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        print('\nTime epoch: ', time.time()-start_time)
        # val_loss, metrics = test_epoch(val_loader, model, loss_fn, device, metrics)
        # val_loss /= len(val_loader)

        # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
        #                                                                          val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            if exp is not None:
                exp.log_metric(metric.name(), metric.value())

        print(message)

        # Neptune logs
        if exp is not None:
            exp.log_metric('epoch_loss', train_loss)
            # exp.log_metric('valid_loss', val_loss)
            exp.log_metric('learning_rate', optimizer.param_groups[0]['lr'])

        ## Save model
        if epoch % args.save_after == 0:
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, f'{args.save_dir}/siamese_model_{epoch}.pt')
            args.load_checkpoint_dir = f'{args.save_dir}/siamese_model_{epoch}.pt'
        # Delete old ones, save latest, keep every 10th
        if (epoch - 1) % 10 != 0:
            try:
                os.remove(f'{args.save_dir}/siamese_model_{epoch - 1}.pt')
            except:
                print("not enough models there yet, nothing to delete")


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics, scaler):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (img1, img2, target, _, _) in enumerate(train_loader):
        img1 = img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # target = target if len(target) > 0 else None
        # if not type(data) in (tuple, list):
        #     data = (data,)

        # data = tuple(d.to(device) for d in data)
        # if target is not None:
        #     target = target.to(device)

        optimizer.zero_grad()
        with amp.autocast():
            outputs = model(img1, img2)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(img1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(f'\r{message}', end="")
            losses = []

    total_loss /= len(train_loader)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (img1, img2, target, _, _) in enumerate(val_loader):
            # target = target if len(target) > 0 else None
            img1 = img1.cuda(non_blocking=True)
            img2 = img2.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            data = (img1, img2)
            # if not type(data) in (tuple, list):
                # data = (data,)
            # if cuda:
                # data = tuple(d.cuda() for d in data)
                # if target is not None:
                    # target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

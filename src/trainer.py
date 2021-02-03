import os
import numpy as np
import torch
from torch.cuda import amp
import time



def fit(args, train_loader, model, loss_fn, optimizer, scheduler,
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

        print('\nTime epoch: ', time.time()-start_time)

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            if exp is not None:
                exp.log_metric(metric.name(), metric.value())

        print(message)

        # Neptune logs
        if exp is not None:
            exp.log_metric('epoch_loss', train_loss)
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

    for batch_idx, (inputs, target, _, _) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        img1, img2 = torch.split(inputs, [3, 3], dim=1)
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



def finetune(args, model, train_loader, valid_loader, loss_fn, optimizer,
             scheduler, exp):
    # Freeze the encoder, train classification head
    model.train()
    scaler = amp.GradScaler()


    for epoch in range(args.start_epoch, args.n_epochs):
        start_time = time.time()
        total_loss, total_num, total_acc = train_bar = 0.0, 0, 0.0, tqdm(train_dataloader)

        # Train stage
        for i, data in enumerate(train_dataloader):
            inputs = data[0].cuda(non_blocking=True)
            target = data[1].cuda(non_blocking=True)

            # Forward pass
            optimizer.zero_grad()

            with amp.autocast():
                # Do not compute the gradients for the frozen encoder
                output = model(inputs)

                # Take pretrained encoder representations
                loss = loss_fn(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs.data, 1)

            total_acc += torch.sum(preds == labels.data)

            total_num += args.batch_size
            total_loss += loss.item() * args.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
            if exp is not None:
                exp.log_metric('loss', total_loss / total_num)
                exp.log_metric('train_acc', total_acc / total_num)


        scheduler.step()

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        print(message)
        print('\nTime epoch: ', time.time()-start_time)

        # Neptune logs
        if exp is not None:
            exp.log_metric('epoch_loss', total_loss)
            exp.log_metric('epoch_acc', total_acc)
            exp.log_metric('learning_rate', optimizer.param_groups[0]['lr'])

        # Validate
        valid_loss, valid_acc, _ = evaluate(args, model, valid_loader)
        if exp is not None:
            exp.log_metric('valid_loss', valid_loss)
            exp.log_metric('valid_acc', valid_acc)

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


def evaluate(args, model, data_loader):
    model.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)

    all_preds, all_labels, all_slides, all_outputs0, all_outputs1, all_patches  = [], [], [], [], [], []

    with torch.no_grad():
        for data, target, patch_id, slide_id in data_bar:
            data = data.cuda(non_blocking=True)

            out = model(data)
            _, preds = torch.max(out.data, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target)
            all_patches.extend(patch_id)
            all_slides.extend(slide_id)

            probs = torch.nn.functional.softmax(out.data, dim=1).cpu().numpy()
            all_outputs0.extend(probs[:, 0])
            all_outputs1.extend(probs[:, 1])

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description(f'Test ACC: {total_correct / total_num * 100:.2f}% ')


    df =  pd.DataFrame({
                'label': all_labels,
                'prediction': all_preds,
                'slide_id': all_slides,
                'patch_id': all_patches,
                'probabilities_0': all_outputs0,
                'probabilities_1': all_outputs1,
            })

    return total_loss / total_num, total_correct / total_num * 100, df
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from . import utils

import gc


def train_fn(
    epoch,
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    config,
    scaler
):
    losses = utils.AverageMeter()

    optimizer.zero_grad()

    # switch to train model
    model.train()

    tk0 = tqdm(
        train_loader,
        total=len(train_loader)
    )

    for step, (images, targets) in enumerate(tk0):
        images = images.to('cuda')
        targets = targets.to('cuda')

        batch_size = targets.size(0)

        with autocast():
            _, loss = model(images, targets, criterion)

        loss = loss / config.n_accumulate

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()

        if (step + 1) % config.n_accumulate == 0:
            if config.max_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad)

            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            optimizer.zero_grad()

        tk0.set_postfix(Epoch=epoch, train_loss=losses.avg)

    gc.collect()

    return losses.avg


def valid_fn(valid_loader, model, criterion, config):
    losses = utils.AverageMeter()
    model.eval()

    preds = []


    tk0 = tqdm(
        valid_loader,
        total=len(valid_loader)
    )
    for step, (images, targets) in enumerate(tk0):
        images = images.to("cuda")
        targets = targets.to("cuda")
        batch_size = targets.size(0)

        with torch.no_grad():
            with autocast():
                logits, loss = model(images, targets, criterion)

        losses.update(loss.item(), batch_size)

        tk0.set_postfix(valid_loss=losses.avg)

        preds.extend(logits.sigmoid().squeeze().cpu().detach().numpy())

    
    return losses.avg, np.array(preds)



def infer_fn(test_loader, model):
    model.eval()

    preds = []

    for step, images in tqdm(enumerate(test_loader)):
        images = images.to("cuda")

        with torch.no_grad():
            with autocast():
                logits = model(images)

        preds.extend(logits.sigmoid().squeeze().cpu().detach().numpy())

    
    return np.array(preds)
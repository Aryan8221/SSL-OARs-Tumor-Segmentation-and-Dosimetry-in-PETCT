import os
import shutil
import time
from logger import setup_logger

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, logger):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    device = torch.device(f'cuda:{args.gpu}')  # Primary device for DataParallel
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"], batch_data["label"]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)  # calculating MSE loss function
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            logger.info(
                f"Epoch {epoch}/{args.max_epochs} {idx}/{len(loader)}\tloss: {run_loss.avg:.4f}\ttime {time.time() - start_time:.2f}s"
            )
        start_time = time.time()
    return run_loss.avg

def val_epoch(model, loader, epoch, args, logger, model_inferer=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    device = torch.device(f'cuda:{args.gpu}')  # Primary device for DataParallel
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(device), target.to(device)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            mse = torch.nn.functional.mse_loss(logits, target, reduction='mean').item()
            if args.distributed:
                mse_list = distributed_all_gather([mse], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_acc.update(np.mean(np.stack(mse_list, axis=0), axis=0), n=1)
            else:
                run_acc.update(mse, n=1)

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                logger.info(
                    f"Val {epoch}/{args.max_epochs} {idx}/{len(loader)}\tmse: {avg_acc}\ttime {time.time() - start_time:.2f}s"
                )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch, args, logger, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    logger.info(f"Saving checkpoint {filename}")


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    args,
    logger,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            logger.info(f"Writing Tensorboard logs to {args.logdir}")
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = float('inf')  # For MSE, lower is better
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        logger.info(f'{args.rank}\t{time.ctime()}\tEpoch: {epoch}')
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, logger=logger
        )
        if args.rank == 0:
            logger.info(
                f"Final training  {epoch}/{args.max_epochs - 1}\tloss: {train_loss:.4f}\ttime {time.time() - epoch_time:.2f}s"
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                args=args,
		logger=logger,
                model_inferer=model_inferer,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                logger.info(
                    f"Final validation  {epoch}/{args.max_epochs - 1}\tmse: {val_avg_acc}\ttime {time.time() - epoch_time:.2f}s"
                )
                if writer is not None:
                    writer.add_scalar("val_mse", val_avg_acc, epoch)
                if val_avg_acc < val_acc_max:  # For MSE, lower is better
                    logger.info("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, logger,  best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, logger, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    logger.info("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    logger.info(f"Training Finished !, Best MSE: {val_acc_max}")

    return val_acc_max

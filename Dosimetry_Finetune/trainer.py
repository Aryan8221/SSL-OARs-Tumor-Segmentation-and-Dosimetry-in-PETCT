import torch.nn.parallel
import torch.utils.data.distributed
import os
import shutil
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from sklearn.metrics import mean_absolute_error, r2_score


def calculate_mape_ignore_zeros(y_true, y_pred):

    mask = y_true != 0

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

    return mape


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, logger):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_mape = AverageMeter()
    run_mae = AverageMeter()
    run_r2 = AverageMeter()
    device = torch.device(f'cuda:{args.gpu}')
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

        logits_np = logits.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        for logit, tgt in zip(logits_np, target_np):
            run_mape.update(calculate_mape_ignore_zeros(tgt.flatten(), logit.flatten()))
            run_mae.update(mean_absolute_error(tgt.flatten(), logit.flatten()))
            run_r2.update(r2_score(tgt.flatten(), logit.flatten()))

        if args.rank == 0:
            logger.info(
                f"Epoch {epoch + 1}/{args.max_epochs} {idx + 1}/{len(loader)}\tRMSE: {np.sqrt(run_loss.avg):.4f}\tMAPE: {run_mape.avg:.4f}\tMAE: {run_mae.avg:.4f}\tR2: {run_r2.avg:.4f}\ttime {time.time() - start_time:.2f}s"
            )
        start_time = time.time()

    return np.sqrt(run_loss.avg), run_mape.avg, run_mae.avg, run_r2.avg


def val_epoch(model, loader, epoch, args, logger, model_inferer=None):
    model.eval()
    run_rmse = AverageMeter()
    run_mape = AverageMeter()
    run_mae = AverageMeter()
    run_r2 = AverageMeter()
    start_time = time.time()
    device = torch.device(f'cuda:{args.gpu}')
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(device), target.to(device)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            logits_np = logits.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            # Convert back to torch tensors for loss calculation
            logits_tensor = torch.tensor(logits_np).to(device)
            target_tensor = torch.tensor(target_np).to(device)

            mse = torch.nn.functional.mse_loss(logits_tensor, target_tensor, reduction='mean').item()
            rmse = np.sqrt(mse)

            if args.distributed:
                rmse_list = distributed_all_gather([rmse], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_rmse.update(np.mean(np.stack(rmse_list, axis=0), axis=0), n=1)
            else:
                run_rmse.update(rmse, n=1)

            for logit, tgt in zip(logits_np, target_np):
                run_mape.update(calculate_mape_ignore_zeros(tgt.flatten(), logit.flatten()))
                run_mae.update(mean_absolute_error(tgt.flatten(), logit.flatten()))
                run_r2.update(r2_score(tgt.flatten(), logit.flatten()))

            if args.rank == 0:
                avg_rmse = np.mean(run_rmse.avg)
                logger.info(
                    f"Val {epoch + 1}/{args.max_epochs} {idx + 1}/{len(loader)}\tRMSE: {avg_rmse:.4f}\tMAPE: {run_mape.avg:.4f}\tMAE: {run_mae.avg:.4f}\tR2: {run_r2.avg:.4f}\ttime {time.time() - start_time:.2f}s"
                )
            start_time = time.time()

    return run_rmse.avg, run_mape.avg, run_mae.avg, run_r2.avg


def save_checkpoint(model, epoch, args, logger, filename="model.pt", best_rmse=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_rmse": best_rmse, "state_dict": state_dict}
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
    best_val_rmse = float('inf')  # For RMSE, lower is better
    for epoch in tqdm(range(start_epoch, args.max_epochs)):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        logger.info(f'{args.rank}\t{time.ctime()}\tEpoch: {epoch}')
        epoch_time = time.time()
        train_rmse, train_mape, train_mae, train_r2 = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, logger=logger
        )
        if args.rank == 0:
            logger.info(
                f"Final training  {epoch + 1}/{args.max_epochs}\tRMSE: {train_rmse:.4f}\tMAPE: {train_mape:.4f}\tMAE: {train_mae:.4f}\tR2: {train_r2:.4f}\ttime {time.time() - epoch_time:.2f}s"
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_rmse", train_rmse, epoch)
            writer.add_scalar("train_mape", train_mape, epoch)
            writer.add_scalar("train_mae", train_mae, epoch)
            writer.add_scalar("train_r2", train_r2, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_rmse, val_mape, val_mae, val_r2 = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                args=args,
                logger=logger,
                model_inferer=model_inferer,
            )

            val_avg_rmse = np.mean(val_avg_rmse)

            if args.rank == 0:
                logger.info(
                    f"Final validation  {epoch + 1}/{args.max_epochs}\tRMSE: {val_avg_rmse:.4f}\tMAPE: {val_mape:.4f}\tMAE: {val_mae:.4f}\tR2: {val_r2:.4f}\ttime {time.time() - epoch_time:.2f}s"
                )
                if writer is not None:
                    writer.add_scalar("val_rmse", val_avg_rmse, epoch)
                    writer.add_scalar("val_mape", val_mape, epoch)
                    writer.add_scalar("val_mae", val_mae, epoch)
                    writer.add_scalar("val_r2", val_r2, epoch)
                if val_avg_rmse < best_val_rmse:  # For RMSE, lower is better
                    logger.info("new best RMSE ({:.6f} --> {:.6f}). ".format(best_val_rmse, val_avg_rmse))
                    best_val_rmse = val_avg_rmse
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, logger, best_rmse=best_val_rmse, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, logger, best_rmse=best_val_rmse, filename="model_final.pt")
                if b_new_best:
                    logger.info("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    logger.info(f"Training Finished !, Best RMSE: {best_val_rmse}")

    return best_val_rmse

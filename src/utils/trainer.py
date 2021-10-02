#  Copyright (c) InstaDeep Ltd. 
#  Licensed under the MIT license.

import os
import time
from bisect import bisect_left
from typing import Tuple

import numpy as np
import psutil
import torch
from torch import nn

from src.utils.training import calculate_metrics


class Trainer:    
    def log_neptune(client, model, optimizer, iters, log_params=False) -> None:
        """
        Logs metrics to neptune
        """
        client["losses/cum_avg_loss_epoch"].log(
            iters.total_loss_epoch / iters.epoch_step)
        client["losses/loss"].log(iters.last_loss)
        client["metric/ppl"].log(iters.last_ppl)
        
        if log_params:    # TODO: iters?
            for param_name, param in model.named_parameters():
                client[f"parameter_mean/{param_name}"].log(
                    param.data.mean())
                client[f"parameter_std/{param_name}"].log(
                    param.data.std())
                if param.grad is None:
                    continue
                client[f"grad_mean/{param_name}"].log(
                    param.grad.data.mean())
                client[f"grad_std/{param_name}"].log(
                    param.grad.data.std())
        client["global/lr"].log(optimizer.param_groups[0]["lr"])
        client["global/memory_usage"].log(psutil.virtual_memory()._asdict()["used"] / 1_000_000)
        client["global/speed"].log(time.time() - iters.last_log)

    @staticmethod
    def evaluate(
        model, 
        dataloader, 
        lm_loss_fn, 
        epoch: int, 
        args
    ) -> Tuple[float, float]:
        """
        
        """
        model.eval()
        tot_loss = []
        tot_ppl = []
        tot_sample = []

        with torch.no_grad():
            for batch in dataloader:
                if args.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{args.local_rank}") for t in batch)
                input_ids, position_ids, token_ids, label_ids, *_ = batch
                n_sample = input_ids.shape[0]

                # TODO: Investigate ppl here and tot_sample
                output = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    token_type_ids=token_ids,
                    labels=None,
                    return_dict=True
                )
                logits = output.logits
                loss, ppl = calculate_metrics(
                    logits=logits,
                    labels=label_ids,
                    loss_fn=lm_loss_fn,
                    ignore_index=-100
                )

                tot_loss.append(loss.mean().item() * n_sample)
                tot_ppl.append(ppl.mean().item() * n_sample)
                tot_sample.append(n_sample)
        model.train()
        print(
            f"\n Epoch {epoch}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} " \
            "Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
        return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)

    def save_checkpoint(
        self,
        model: nn.Module, 
        path: str, 
        prefix: str = '', 
        checkpoint_name: str = "checkpoint.pth",
        to_bucket: bool = False,
        bucket=None,
        bucket_dir: str = None,
        cleanup: bool = False
    ):
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.config.save_pretrained(path)
        state_dict = model_to_save.state_dict()
        dest_path = os.path.join(path, prefix + checkpoint_name)
        torch.save(state_dict, dest_path)

        if to_bucket:
            self.upload_to_bucket(
                dest_path, bucket_dir, bucket, cleanup=cleanup
            )

    @staticmethod
    def upload_to_bucket(ckpt: str, dest_dir: str, bucket, cleanup: bool = False):
        filename = os.path.split(ckpt)[1]
        dest_path = dest_dir + "/" + filename
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(ckpt)
        
        if cleanup:
            os.remove(ckpt)

    @staticmethod
    def save_val_ckpt(existing_ckpts, ppl, args):
        idx = bisect_left([v[1] for v in existing_ckpts], ppl)
        if (idx < len(existing_ckpts)) or (len(existing_ckpts) < args.num_val_ckpts):
            return idx
        return -1

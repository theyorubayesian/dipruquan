#  Copyright (c) InstaDeep Ltd. 
#  Licensed under the MIT license. 

import argparse
import logging
import os
import time

import torch
from google.cloud import storage
from torch import nn
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader

from src.msft.data_loader import BucketingDataLoader
from src.msft.data_loader import DistributedBucketingDataLoader
from src.utils.checks import _run_sanity_checks
from src.utils.eval import generate_responses
from src.utils.model import get_models
from src.utils.parser import parse_general_args
from src.utils.trainer import Trainer
from src.utils.training import Args
from src.utils.training import calculate_metrics
from src.utils.training import init_gpu_params
from src.utils.training import set_seed
from src.utils.training import set_lr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_other_args(extras, args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from_hf", 
        action="store_true", 
        help="Get model from HuggingFace Model Hub"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model to get from HF Model Hub"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model config.json file"
    )
    parser.add_argument(
        "--model_pretrained_weights",
        type=str,
        help="Path to model pretrained checkpoint"
    )
    parser.add_argument(
        "--generate_responses", 
        action="store_true", 
        help="Generate responses on test set after every epoch"
    )
    parser.add_argument(
        "--num_val_ckpts",
        default=-1,
        type=int,
        help="Number of validation checkpoints (N) to store. The first `N` are always stored. "
    )
    
    args = parser.parse_args(extras, args)
    return args


class DialoGPTTrainer(Trainer):
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer, 
        optimizer, 
        logging_client, 
        bucket, 
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        clm_loss_fn,
        *args
    ) -> None:
        self.model = model,
        self.tokenizer = tokenizer,
        self.optimizer = optimizer,
        self.client = logging_client,
        self.bucket = bucket,
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.clm_loss_fn = clm_loss_fn

    def train(self, args: dict):
        if args.is_master:
            logger.info("Starting training")
        
        self.model.train()
        iters = Args(
            {
                "days":  0,
                "epoch": 0,
                "epoch_step": 0,
                "global_step": 0,
                "step": 0,
                "total_training_time": 0,
                "last_loss": 0,
                "last_ppl": 0,
                "total_loss_epoch": 0,
                "n_sequences_epoch": 0,
                "last_log": time.time(),
                "last_iter_time": time.time(),
                "validation_ckpts": []
            }
        )
        if args.continue_from:
            iters.epoch = args.continue_from-1
            iters.global_step = int(
                args.continue_from 
                * (args.total_num_sequences / args.batch_size / args.gradient_accumulation_steps)
                )     # Hack 
            iters.step = int(args.continue_from * (args.total_num_sequences / args.batch_size)) 
        
        while iters.epoch < args.n_epoch:
            if args.multi_gpu:
                torch.distributed.barrier()

            for batch in self.train_dataloader:
                if args.n_gpu > 0:
                    batch = tuple(t.to(args.device) for t in batch)
                input_ids, position_ids, token_ids, label_ids, *_ = batch

                output = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    token_type_ids=token_ids,
                    labels=None,
                    return_dict=True,
                    output_hidden_states=False
                )
                logits = output.logits
                
                loss, ppl = calculate_metrics(
                    logits=logits,
                    labels=label_ids,
                    loss_fn=self.clm_loss_fn,
                    ignore_index=-100
                )
                
                if (loss != loss).data.any():
                    logger.error("NaN detected in loss")
                    exit()
                
                if args.multi_gpu:
                    loss = loss.mean()
                    ppl = ppl.mean()
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                
                iters.last_ppl = ppl.item()
                iters.last_loss = loss.item()
                iters.total_loss_epoch += loss.item()
                iters.step += 1
                iters.epoch_step += 1
                iters.total_training_time += time.time() - iters.last_iter_time
                iters.last_iter_time = time.time()
                
                if iters.step % args.gradient_accumulation_steps == 0:
                    iters.global_step += 1
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.max_grad_norm
                    )
                    self.optimizer.step()
                    if args.is_master:
                        self.log_neptune(self.client, self.model, self.optimizer, iters)
                        args.last_log = time.time()
                    
                    set_lr(
                        [self.optimizer],
                        iters.global_step,
                        args.learning_rate,
                        args.warmup_steps,
                        args.n_embd
                    )
                    self.optimizer.zero_grad()
                    
                    if args.is_master:
                        if iters.global_step % args.checkpoint_interval == 0:
                            self.save_checkpoint(
                                self.model, 
                                path=args.output_dir, 
                                checkpoint_name=f"{iters.global_step}_checkpoint.pth",
                                to_bucket=args.to_bucket,
                                bucket=self.bucket,
                                bucket_dir=args.bucket_dir,
                                cleanup=args.cleanup
                            )

                        if args.validate and \
                            (iters.global_step % args.validation_interval == 0):
                            eval_loss, eval_ppl = self.evaluate(
                                model=self.model,
                                dataloader=self.eval_dataloader,
                                lm_loss_fn=self.clm_loss_fn,
                                epoch=iters.epoch,
                                args=args
                            )
                            self.client["val/loss"].log(eval_loss)
                            self.client["val/ppl"].log(eval_ppl)

                            idx = self.save_val_ckpt(iters.validation_ckpts, eval_ppl, args)

                            if idx != -1:
                                iters.validation_ckpts.insert(idx, (iters.global_step, eval_ppl))
                                if len(iters.validation_ckpts) > args.num_val_ckpts:
                                    iters.validation_ckpts.pop()

                                self.save_checkpoint(
                                    self.model,
                                    path=args.output_dir,
                                    checkpoint_name=f"{iters.global_step}_val_checkpoint.pth",
                                    to_bucket=args.to_bucket,
                                    bucket=self.bucket,
                                    bucket_dir=args.bucket_dir,
                                    cleanup=args.cleanup
                                )
                
                if args.is_master:
                    if iters.total_training_time // 86400 > iters.days:
                        iters.days += 1
                        self.save_checkpoint(
                            self.model, 
                            args.output_dir,
                            checkpoint_name=f"model_day_{iters.days}.pth",
                            to_bucket=args.to_bucket,
                            bucket=self.bucket,
                            bucket_dir=args.bucket_dir,
                            cleanup=args.cleanup
                        )
                
                iters.n_sequences_epoch += input_ids.size(0)
            # iters.n_toten_total += input_ids.shape[0] * input_ids.shape[1]
            # iters.n_token_real += (input_ids != 0).sum().item()
            
            if args.is_master:
                logger.info(f"Ending epoch {iters.epoch}/{args.n_epoch-1}")
                self.save_checkpoint(
                    self.model, path=args.output_dir, 
                    checkpoint_name=f"model_epoch_{iters.epoch}.pth", 
                    to_bucket=args.to_bucket,
                    bucket=self.bucket,
                    bucket_dir=args.bucket_dir,
                    cleanup=args.cleanup
                )  # TODO
                self.client["epoch/loss"].log(iters.total_loss_epoch / iters.epoch_step)

                if args.generate_responses:
                    self.model.eval()

                    generate_responses(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        batch_size=64,
                        tokenizer_max_len=128,
                        model_max_len=128,
                        beam=1,
                        context_file="compression/distillation/msft/dstc/data/test.source",
                        output_file=f"{args.output_dir}/model.epoch_{iters.epoch}.6k.resp.txt"
                    )   # TODO
                    
                    self.model.train()
            
            iters.epoch += 1
            iters.epoch_step = 0
            iters.n_sequences_epoch = 0
            iters.total_loss_epoch = 0
        
        if args.is_master:
            eval_loss, eval_ppl = self.evaluate(
                                model=self.model,
                                dataloader=self.eval_dataloader,
                                lm_loss_fn=self.clm_loss_fn,
                                epoch=iters.epoch,
                                args=args
                            )
            self.client["val/loss"].log(eval_loss)
            self.client["val/ppl"].log(eval_ppl)

            logger.info("Saving final checkpoint as `pytorch_model.bin`")
            self.save_checkpoint(self.model, path=args.output_dir, checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")


def main():
    args, extras = parse_general_args()
    args = parse_other_args(extras, args)

    init_gpu_params(args)
    _run_sanity_checks(args)
    set_seed(args)
    bucket=None
    client = None
    args.device = f"cuda:{args.local_rank}"

    if args.is_master:
        import neptune.new as neptune
        from dotenv import load_dotenv

        load_dotenv()
        client = neptune.init(
            project=os.getenv("NEPTUNE_USERNAME") + os.getenv("NEPTUNE_PROJECT_NAME"),
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            mode=os.getenv("NEPTUNE_CONNECTION_MODE"),
            source_files=[]
        )
        args_copy = vars(args)
        for a in args_copy:
            client[a] = args_copy[a]
            logger.info(f"{a}:  {args_copy[a]}")

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(os.getenv("GOOGLE_CLOUD_BUCKET_NAME"))
    
    tokenizer, model, _ = get_models(
        from_hf=args.from_hf,
        get_student=False,
        model_name=args.model_name,
        teacher_config_json=args.model_config,
        teacher_pretrained_weights=args.model_pretrained_weights,
        student_config_json=None
    )
    args.n_embd = model.config.n_embd
    
    clm_loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    
    if args.n_gpu > 0:
        model.to(args.device)
    
    logger.info(f"Model location: {model.device}")
    
    if args.multi_gpu:
        from torch.nn.parallel import DistributedDataParallel
        
        logger.info("Using nn.parallel.DistributedDataParallel for distributed training")
        
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
        dataloader = DistributedBucketingDataLoader(
            get_rank(),
            get_world_size(),
            args.data,
            args.batch_size,
            args.max_seq_length
        )
    else:
        dataloader = BucketingDataLoader(
            args.data,
            args.batch_size,
            args.max_seq_length
        )
    
    eval_dataloader = None
    if args.is_master:
        eval_dataloader = BucketingDataLoader(
            args.eval_data,
            args.eval_batch_size,
            args.max_seq_length
        )

    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay) and p.requires_grad
            ], 
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            betas=(0.9, 0.999)
        )

    torch.cuda.empty_cache()
    
    trainer = DialoGPTTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        logging_client=client,
        bucket=bucket,
        train_dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        clm_loss_fn=clm_loss_fn
    )

    logger.info("Let's go get some drinks")
    trainer.train(args=args)


if __name__ == "__main__":
    main()

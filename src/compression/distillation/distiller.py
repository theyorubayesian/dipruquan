import logging
import os
import time
from bisect import bisect_left
from typing import Tuple

import numpy as np
import psutil
import torch
from google.cloud import storage
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.utils.eval import generate_responses
from src.utils.trainer import Trainer
from src.utils.training import calculate_metrics
from src.utils.training import set_lr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Distiller(Trainer):
    """
    Source: Distil*
    https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation
    """
    def __init__(
        self,
        teacher,
        student,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        params
    ) -> None:
        logger.info("Initializing Distiller")
        self.student = student
        self.teacher = teacher
        self.n_embd = teacher.config.n_embd
        self.dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.params = params

        self.temperature = params.temperature
        self.alpha_clm = params.alpha_clm
        self.alpha_ce = params.alpha_ce
        self.alpha_cos = params.alpha_cos

        self.is_master = params.is_master
        self.multi_gpu = params.multi_gpu

        if self.is_master:
            import neptune.new as neptune
            from dotenv import load_dotenv

            load_dotenv()
            self.client = neptune.init(
                project=f"a.oladipo/" + os.getenv("NEPTUNE_PROJECT_NAME"),
                api_token=os.getenv("NEPTUNE_API_TOKEN"),
                mode=os.getenv("NEPTUNE_CONNECTION_MODE"),
                source_files=[]
            )
            args_copy = vars(params)
            for a in args_copy:
                self.client[a] = args_copy[a]
                logger.info(f"{a}:  {args_copy[a]}")
            
            storage_client = storage.Client()
            self.bucket = storage_client.get_bucket(os.getenv("GOOGLE_CLOUD_BUCKET_NAME"))
            
            self.tokenizer = tokenizer

        self.days = 0
        self.epoch = 0
        self._step = 0
        self.epoch_step = 0
        self.global_step = 0
        self.n_sequences_epoch = 0
        self.last_log = 0
        self.last_iter_time = 0
        self.total_training_time = 0

        self.s_last_loss = 0
        self.s_last_loss_ce = 0
        self.s_last_loss_clm = 0
        self.s_last_loss_cos = 0
        self.s_last_ppl = 0
        self.s_total_loss_epoch = 0

        self.ce_loss_fn = nn.KLDivLoss(reduction="mean")
        self.lm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100, reduction="none")
        self.cosine_loss_fn = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("Initializing optimizers")
        no_decay = ["bias", "ln"]
        s_optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0
            },
        ]

        s_trainable_params = sum(
            [p.numel() for p in self.student.parameters() if p.requires_grad])
        s_params = sum([p.numel() for p in self.student.parameters()])
        t_params = sum([p.numel() for p in self.teacher.parameters()])
        
        logger.info(
            "Number of trainable parameters (student): %i" % s_trainable_params)
        logger.info(
            "Number of parameters (student): %i" % s_params
        )
        logger.info("Number of parameters (teacher): %i" % t_params)
        
        self.student_optimizer = AdamW(
            s_optimizer_grouped_parameters,
            lr=params.learning_rate,
            eps=params.adam_epsilon,
            betas=(0.9, 0.999)
        )
        
        if self.multi_gpu:
            from torch.nn.parallel import DistributedDataParallel
            
            logger.info("Using nn.parallel.DistributedDataParallel for distributed training")
            
            self.student = DistributedDataParallel(
                self.student,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                find_unused_parameters=True
            )
    
    def train(self) -> None:
        if self.is_master:
            logger.info("Starting training")
        self.last_iter_time = time.time()
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.multi_gpu:
                torch.distributed.barrier()

            for batch in self.dataloader:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                input_ids, position_ids, token_ids, label_ids, *_ = batch

                self.step(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    token_ids=token_ids,
                    lm_labels=label_ids
                )
            if self.is_master:
                logger.info(f"Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Saving final checkpoint as `pytorch_model.bin`")
            self.save_checkpoint(self.student, path=self.params.output_dir, heckpoint_name="pytorch_model.bin")
            logger.info("Training is finished")
    
    def step(
        self,
        input_ids: torch.tensor,
        position_ids: torch.tensor,
        token_ids: torch.tensor,
        lm_labels: torch.tensor
    ) -> None:
        s_output = self.student(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_ids,
            labels=None,
            return_dict=True,
            output_hidden_states=True
        )
        s_logits = s_output.logits
        s_hidden_states = s_output.hidden_states[-1]

        s_loss_clm, s_ppl = calculate_metrics(
            logits=s_logits,
            labels=lm_labels,
            loss_fn=self.lm_loss_fn,
            ignore_index=-100
        )
        self.s_last_ppl = s_ppl.item()
        s_loss = self.alpha_clm * s_loss_clm
        
        with torch.no_grad():
            t_output = self.teacher(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_ids,
                labels=None,
                return_dict=True,
                output_hidden_states=True
            )
            t_logits = t_output.logits
            t_hidden_states = t_output.hidden_states[-1]
        
        assert s_logits.size() == t_logits.size()
        
        mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)   # (bs, seq_length, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
        t_logits_slct = torch.masked_select(t_logits, mask)
        t_logits_slct = t_logits_slct.view(-1, t_logits.size(-1))

        assert t_logits_slct.size() == s_logits_slct.size()
        
        s_logits_sft = F.log_softmax(s_logits_slct / self.temperature, dim=-1)
        t_logits_sft = F.softmax(t_logits_slct / self.temperature, dim=-1)

        s_loss_ce = (
            self.ce_loss_fn(s_logits_sft, t_logits_sft) *
            (self.temperature ** 2)
        )
        s_loss += self.alpha_ce * s_loss_ce
        
        # s_hidden_states = s_hidden_states[-1]   # (bs, seq_length, dim)
        # t_hidden_states = t_hidden_states[-1]

        dim = s_hidden_states.size(-1)
        # s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)
        s_hidden_states_slct = s_hidden_states.view(-1, dim)

        # t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)
        t_hidden_states_slct = t_hidden_states.view(-1, dim)

        target = s_hidden_states_slct.new(
            s_hidden_states_slct.size(0)).fill_(1)
        s_loss_cos = self.cosine_loss_fn(
            s_hidden_states_slct, t_hidden_states_slct, target)
        s_loss += self.alpha_cos * s_loss_cos

        self.s_last_loss = s_loss.item()
        self.s_last_loss_ce = s_loss_ce.item()
        self.s_last_loss_clm = s_loss_clm.item()
        self.s_last_loss_cos = s_loss_cos.item()
        self.s_total_loss_epoch += s_loss.item()
        
        self.optimize(s_loss)
        
        self.n_sequences_epoch += input_ids.size(0)
    
    def optimize(self, s_loss) -> None:
        if (s_loss != s_loss).data.any():
            logger.error("NaN detected in student loss")
            exit()

        if self.multi_gpu:
            s_loss = s_loss.mean()
            
        if self.params.gradient_accumulation_steps > 1:
            s_loss = s_loss / self.params.gradient_accumulation_steps

        s_loss.backward()

        self.iter()
        if self._step % self.params.gradient_accumulation_steps == 0:
            self.global_step += 1
            nn.utils.clip_grad_norm_(
                self.student.parameters(), self.params.max_grad_norm)

            self.student_optimizer.step()

            self.log_neptune()
            self.last_log = time.time()
            
            set_lr(
                [self.student_optimizer],
                self.global_step,
                self.params.learning_rate,
                self.params.warmup_steps,
                self.n_embd
            )
            self.student_optimizer.zero_grad()

            if self.global_step and \
                (self.global_step % self.params.validation_interval == 0):
                if self.is_master and self.params.validate:
                    eval_loss, eval_ppl = self.evaluate(
                        model=self.student,
                        dataloader=self.eval_dataloader,
                        epoch=self.epoch
                    )
                    self.client["student/val/loss"].log(eval_loss)
                    self.client["student/val/ppl"].log(eval_ppl)
                    
                    idx = self.save_val_ckpt(self.validation_ckpts, eval_ppl, self.params)
                    
                    if idx != -1:
                        self.validation_ckpts.insert(idx, (self.global_step, eval_ppl))
                        if len(self.validation_ckpts) > self.params.num_val_ckpts:
                            self.validation_ckpts.pop()
                    
                        self.save_checkpoint(
                            self.student,
                            path=self.params.output_dir,
                            checkpoint_name=f"{self.global_step}_val_ckpt.distill.pth",
                            to_bucket=self.params.to_bucket,
                            bucket=self.bucket,
                            bucket_dir=self.params.bucket_dir,
                            cleanup=self.params.cleanup
                        )
            torch.cuda.empty_cache()
    
    def iter(self) -> None:
        self._step += 1
        self.epoch_step += 1
        self.total_training_time += time.time() - self.last_iter_time
        self.last_iter_time = time.time()
        
        if self.is_master:
            if self.global_step and \
            (self.global_step % self.params.checkpoint_interval == 0):
                self.save_checkpoint(
                    self.student,
                    path=self.params.output_dir,
                    checkpoint_name=f"{self.global_step}_ckpt.distill.pth",
                    to_bucket=self.params.to_bucket,
                    bucket=self.bucket,
                    bucket_dir=self.params.bucket_dir,
                    cleanup=self.params.cleanup
                )
            
            if self.total_training_time // 86400 > self.days:
                self.days += 1
                self.save_checkpoint(
                    self.student,
                    path=self.params.output_dir,
                    checkpoint_name=f"model_day_{self.days}.distill.pth",
                    to_bucket=self.params.to_bucket,
                    bucket=self.bucket,
                    bucket_dir=self.params.bucket_dir,
                    cleanup=self.params.cleanup
                )

    def log_neptune(self) -> None:
        if not self.is_master:
            return

        self.client["student/losses/cum_avg_loss_epoch"].log(
            self.s_total_loss_epoch / self.epoch_step)
        self.client["student/losses/loss"].log(self.s_last_loss)
        self.client["student/losses/loss_ce"].log(self.s_last_loss_ce)
        self.client["student/losses/loss_clm"].log(self.s_last_loss_clm)
        self.client["student/losses/loss_cos"].log(self.s_last_loss_cos)
        self.client["student/ppl"].log(self.s_last_ppl)

        self.client["global/student/lr"].log(self.student_optimizer.param_groups[0]["lr"])
        self.client["global/memory_usage"].log(psutil.virtual_memory()._asdict()["used"] / 1_000_000)
        self.client["global/speed"].log(time.time() - self.last_log)
        
        if self.params.log_model_params:
            for param_name, param in self.student.named_parameters():
                self.client[f"student/parameter_mean/{param_name}"].log(
                    param.data.mean())
                self.client[f"student/parameter_std/{param_name}"].log(
                    param.data.std())
                if param.grad is None:
                    continue
                self.client[f"student/grad_mean/{param_name}"].log(
                    param.grad.data.mean())
                self.client[f"student/grad_std/{param_name}"].log(
                    param.grad.data.std())
    
    def end_epoch(self) -> None:
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")
        if self.is_master:
            self.save_checkpoint(
                self.student,
                path=self.params.output_dir,
                checkpoint_name=f"model_epoch_{self.epoch}.distill.pth",
                to_bucket=self.params.upload_ckpt_to_bucket,
                bucket=self.bucket,
                bucket_dir=self.params.bucket_dir,
                cleanup=self.params.cleanup
            )
            self.client["student/epoch/loss"].log(self.s_total_loss_epoch / self.epoch_step)
            
            if self.params.generate_responses:
                self.student.eval()
                generate_responses(
                    model=self.student,
                    tokenizer=self.tokenizer,
                    batch_size=64,
                    tokenizer_max_len=128,
                    model_max_len=128,
                    beam=1,
                    context_file="src/msft/dstc/data/test.source",    # TODO
                    output_file=f"{self.params.output_dir}/student.model.epoch_{self.epoch}.6k.resp.txt"
                )
                self.student.train()
        self.epoch += 1
        self.epoch_step = 0
        self.n_sequences_epoch = 0
        self.s_total_loss_epoch = 0
 
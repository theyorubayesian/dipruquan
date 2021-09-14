import logging
import os
import socket

import numpy as np
import torch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Args:
        def __init__(self, args_dict):
            for k, v in args_dict.items():
                setattr(self, k, v)


def noam_decay(step, warmup_steps, model_size):
    return (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


def set_lr(
    optimizer: list,
    step: int,
    lr: float,
    warmup_steps: int,
    n_embd: int,
    has_masked_scores_params: bool = False,
    masked_lr: float = None
):
    lr_this_step = lr * 1e4 * noam_decay(step, warmup_steps, n_embd)
    
    for opt in optimizer:
        for i, param_group in enumerate(opt.param_groups):
            if (i == 0) and has_masked_scores_params:
                lr_this_step = masked_lr * 1e4 * noam_decay(step, warmup_steps, n_embd)
            param_group['lr'] = lr_this_step


def calculate_metrics(logits, labels, loss_fn, ignore_index: int = -100):
    """Calculates CLM loss and perplexity"""
    loss1 = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss1 = loss1.view(labels.size(0), labels.size(1))
    label_size = torch.sum(labels != ignore_index, dim=1).type(loss1.type())
    loss = torch.sum(loss1) / torch.sum(label_size)
    ppl = torch.exp(
        torch.mean(
            torch.sum(loss1, dim=1).float() / label_size.float()
        )
    )
    return loss, ppl


def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        

def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    Source: https://github.com/microsoft/DialoGPT
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])
        params.is_master = params.node_id == 0 and params.local_rank == 0
    # local job (single GPU)
    else:
        assert params.local_rank in [-1, 0, 1]

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0 if params.local_rank == -1 else params.local_rank
        params.global_rank = 1
        params.world_size = 2
        params.n_gpu_per_node = 2
        params.multi_gpu = False
        params.is_master = True

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )
    
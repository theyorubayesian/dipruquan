#  Copyright (c) InstaDeep Ltd. 
#  Licensed under the MIT license.

import argparse


def parse_general_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Input data dir"
    )
    parser.add_argument(
        "--total_num_sequences",
        type=int,
        default=653720,
    ) 
    parser.add_argument(
        "--eval_data",
        default=None,
        type=str,
        required=True,
        help="Validation data dir"
    )
    parser.add_argument(
        "--dump-path",
        type=str,
        required=True,
        help="Output directory where predictions & ckpts will be written"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite content of output directory"
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before backward/update pass"
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="Initial learning rate for Adam")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if applied")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument(
            "--n_epochs",
            default=5,
            type=int,
            help="Total number of training epochs to perform"
        )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps"
    )
    parser.add_argument(
        "--num-optim-steps",
        default=-1,
        type=int,
        help="If > 0, overrides num_training_epochs and set no of training steps to perform"
    )
    parser.add_argument(
        "--checkpoint-interval",
        default=5000,
        type=int,
        help="Interval to save model checkpoint"
    )
    parser.add_argument(
        "--validation-interval",
        default=8,
        type=int,
        help="Interval to perform validation"
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Max total input sequence length after tokenization. "
        "Truncation & Padding occurs for longer/shorter sentences respectively"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization"
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="For distributed training: local_rank"
    )
    parser.add_argument("--upload_ckpt_to_bucket", action="store_true")
    parser.add_argument("--bucket_dir", type=str, help="Path on GCP Bucket to store ckpts")
    parser.add_argument(
        "--cleanup", 
        action="store_true", 
        help="Clean up Checkpoint files after upload to GCP Bucket"
    )

    args, extras = parser.parse_known_args()
    return args, extras

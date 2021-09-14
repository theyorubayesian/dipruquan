#  Copyright (c) InstaDeep Ltd. 
#  Licensed under the MIT license. 

import os
import shutil


def _run_sanity_checks(args):
    if args.is_master:
        if hasattr(args, "output_dir") and os.path.exists(args.output_dir):
            if not args.overwrite_output_dir:
                if not args.force:
                    raise ValueError(
                        f"Serialization dir {args.dump_path} already exists, but you have not precised wheter to overwrite it"
                        "Use `--force` if you want to overwrite it"
                    )
                else:
                    shutil.rmtree(args.dump_path)
        else:
            os.makedirs(args.dump_path)
    
    if args.from_hf:
        if not args.model_name:
            raise ValueError(
                "`model-name` is required to load model from HuggingFace")
    
    assert args.gradient_accumulation_steps >= 1

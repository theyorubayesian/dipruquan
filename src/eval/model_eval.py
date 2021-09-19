#  Copyright (c) InstaDeep Ltd. 
#  Licensed under the MIT license. 

import logging
import os
import subprocess
import time

import click
import torch
from nltk.tokenize import word_tokenize
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def fix_state_dict_namespace(model_state_dict, prefix: str = "transformer."):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith(prefix):
            new_key = t.replace(prefix, '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)
    try:
        model_state_dict.pop("lm_head.decoder.weight")  # TODO: Investigate
    except KeyError:
        pass

    return model_state_dict


def load_assets(model_config_file: str, model_ckpt: str):
    """Load and return model (in eval mode, from checkpoint) and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config.from_pretrained(model_config_file)
    model = GPT2LMHeadModel(config)
    start_model = model

    model_state_dict = torch.load(model_ckpt)
    if model_ckpt.endswith("pkl"):
        model_state_dict = fix_state_dict_namespace(
            model_state_dict, prefix="transformer.")

    if hasattr(model, "transformer") \
            and all(not s.startswith("transformer.") for s in model_state_dict.keys()):
        start_model = model.transformer

    start_model.load_state_dict(model_state_dict, strict=False)
    model.lm_head.weight.data = model.transformer.wte.weight.data
    model = model.eval()

    return model, tokenizer


def generate_responses(
        model,
        tokenizer,
        batch_size,
        tokenizer_max_len,
        model_max_len, 
        beam,
        context_file, 
        output_file
    ):
    """
    Generate a response for each context in context_file.
    Write output to `output_file`
    Adapted from issue #63 by Shilei Liu
    https://github.com/microsoft/DialoGPT/issues/63
    """
    SEP = tokenizer.eos_token

    j = 0
    context_exists = True
    f = open(context_file, 'r')
    out = open(output_file, "w")

    while context_exists:
        i = 0
        batch = []
        while i < batch_size:
            line = f.readline()
            if not line:
                context_exists = False
                break
            context = SEP.join(line.strip().split(' EOS ')) + SEP
            batch.append(context)
            i += 1
            j += 1

        if not batch:
            break

        inputs = tokenizer(batch, max_length=tokenizer_max_len,
                           padding=True, truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        context_length = input_ids.shape[1]

        preds_id = model.generate(
            input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_length=model_max_len,  # TODO: Investigate
            num_beams=beam,
            pad_token_id=tokenizer.eos_token_id
        )
        preds_id = preds_id[:, context_length:].tolist()

        for pred in preds_id:
            response = tokenizer.decode(pred, skip_special_tokens=True)
            out.write(' '.join(word_tokenize(response)) + "\n")

        if j % 512 == 0:
            logger.info(f"{j} responses written to {output_file}")
    out.close()
    f.close()
    logger.info(f"{j} responses written to {output_file}")


@click.command()
@click.option("--model-ckpt")
@click.option("--model-name")
@click.option(
    "--model-config-file", default="src/models/dialogpt/configs/124M/small.json")
@click.option("--from-hf", is_flag=True, help="Get model from HuggingFace.")
@click.option("--context-file", default="src/eval/data/test.source", show_default=True)
@click.option("--output-file", default="outputs/model.6k.resp.txt", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite existing output file")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--tokenizer-max-len", default=128, show_default=True)
@click.option("--model-max-len", default=256, show_default=True)
@click.option("--beam", default=1, show_default=True)
@click.option(
    "--eval-type",
    type=click.Choice(['dstc', 'multiref'], case_sensitive=False),
    default="multiref",
    show_default=True)
@click.option("--refs", default="src/eval/data/test.refs.txt", show_default=True)
@click.option("--keys", default="src/eval/data/keys.6k.txt", show_default=True)
@click.option("--vshuman", default=-1, show_default=True)
def main(
    model_ckpt,
    model_name,
    model_config_file,
    from_hf,
    context_file,
    output_file,
    force,
    batch_size,
    tokenizer_max_len,
    model_max_len,
    beam,
    eval_type,
    refs,
    keys,
    vshuman
):
    if from_hf:
        if not model_name:
            raise ValueError("`model-name` is required to load model from HuggingFace")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        model, tokenizer = load_assets(model_config_file, model_ckpt)
    logger.info("Successfully loaded model and tokenizer")
    
    if model_name:
        model_name: str = model_name.split('/')[-1]  # convert microsoft/Dialogpt-medium to Dialogpt-medium
        output_file = output_file.replace("model", model_name)

    if not force and os.path.exists(output_file):
        raise ValueError("Found existing response file. Please backup.")
    
    start_time = time.time()
    generate_responses(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        tokenizer_max_len=tokenizer_max_len,
        model_max_len=model_max_len,
        beam=beam,
        context_file=context_file,
        output_file=output_file
    )
    logger.info(f"6000 reponses: {(time.time() - start_time)/60} mins")
    
    eval_command = \
        f"""
        python src/eval/dstc.py 
        {output_file} 
        --ref {refs} 
        --keys {keys} 
        --vshuman {vshuman}
        """
    logger.info(f"Starting evaluation.\nEval command:\n{eval_command}")
    eval_result = subprocess.check_output(eval_command.split())
    print(eval_result)


if __name__ == "__main__":
    main()

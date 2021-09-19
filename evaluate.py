import subprocess
import sys

command = \
    """
    python compression/distillation/msft/dstc/model_eval.py
    --model-name microsoft/DialoGPT-medium
    --from-hf
    --context-file compression/distillation/msft/dstc/data/test.source
    --output-file results/model.6k.resp.txt
    --force
    --batch-size 64
    --tokenizer-max-len 128
    --model-max-len 256
    --beam 10
    --refs compression/distillation/msft/dstc/data/test.refs.txt
    --keys compression/distillation/msft/dstc/data/keys.6k.txt
    --vshuman -1
    """

print(command)

p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in iter(p.stdout.readline, b''):
    sys.stdout.write(line.decode(sys.stdout.encoding))

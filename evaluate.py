import subprocess
import sys

command = \
    """
    python src/eval/model_eval.py
    --model-name microsoft/DialoGPT-medium
    --from-hf
    --context-file src/eval/data/test.source
    --output-file outputs/dialogpt-medium.6k.resp.txt
    --force
    --batch-size 64
    --tokenizer-max-len 128
    --model-max-len 256
    --beam 10
    --refs src/eval/data/test.refs.txt
    --keys src/eval/data/keys.6k.txt
    --vshuman -1
    """

print(command)

p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in iter(p.stdout.readline, b''):
    sys.stdout.write(line.decode(sys.stdout.encoding))

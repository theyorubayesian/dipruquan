# dipruQuan

`dipruQuan` explores compresssion methods (`di`stillation, `pru`ning and `Quan`tization) of conversational AI models.
As of 13/09/2021, it contains code for the online and offline distillation of [DialoGPT](https://github.com/microsoft/DialoGPT) 

## Setup
This project has only been tested for `Linux` architectures. 
* Create a conda environment from [LSP-linux.yml](LSP-linux.yml)
* Install any [Pytorch 1.2](https://pytorch.org/get-started/previous-versions/) version compatible with the version of `CUDA Toolkit` on your machine. 
* For mixed precision training, install `Apex`. See [Apex: Quickstart](https://github.com/NVIDIA/apex#linux)

## Data
The training dataset (Reddit small) should be around 140MB while the raw validation dataset should be about 4.7GB. To generate both datasets and the 6K Multi-Reference Datset, run
```
cd src/msft/reddit_extractor
make -j 4
```
This operation may require up to `500GB of local disk space` and will take significant time to complete.

### Training and Validation Set:
The `make` command should create both `data/train_raw.tsv` and `data/validation_raw.tsv`. Both datasets have to be compressed into lazy-loading database files for use in training. Here are the steps:

* Convert the file to the right format
    ```commandline
    cd data
    less train_raw.tsv | awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}'> train.tsv; cd ..
    ```
* Compress into database file
    ```commandline
    python src/msft/prepro.py --corpus data/train.tsv
    ```

### 6K Multi-reference test set:
After running the `make` command, the 6k multi-reference test will be located at `data/test.refs.txt`. You need to create a `test.source` containing the prompt sentences for which the model is to generate responses. 
```commandline
cd data
cat test.refs.txt | cut -f 1 > test.source
mv test.source ../src/eval/data
cat test.refs.txt | cut -f 2- | rev | cut -f 2- | rev > test.refs.tmp.txt
paste keys.6k.txt test.refs.tmp.txt > test.refs.txt
mv test.refs.txt ../src/eval/data
```
See [#48](https://github.com/microsoft/DialoGPT/issues/48) and [#63](https://github.com/microsoft/DialoGPT/issues/63)

## Distillation


## Evaluation
You are going to need a few things:
- [meteval14.pl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl)
- [meteor-1.5](http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz) to compute [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/index.html). It requires Java.

See [3rdparty](src/eval/3rdparty/README.md) for more information about this.

To evaluate any model or checkpoint, use [model_eval.py](src/eval/model_eval.py)
```
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
```
This will generate responses to the `6K multi-ref` and evaluate. Evaluation results will be available in the same directory as the `output-file` specified. 

## Caveats

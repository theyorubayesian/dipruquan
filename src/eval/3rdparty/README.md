# Notes
This folder contains third-party packages needed to calculate some of the evaluation metrics.
Please download the following packages into this folder:
* [meteor-1.5](http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz) to compute [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/index.html). It requires Java.
* [mteval-v14.pl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl) to compute [NIST](http://www.mt-archive.info/HLT-2002-Doddington.pdf)
    * You also need to install some `perl` modules as follows
    ```commandline
    cpan install XML:Twig Sort:Naturally String:Util
    ```
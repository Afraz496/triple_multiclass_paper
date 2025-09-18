# Multiclass ML Pipeline

This repository consists of software supporting the [Comprehensive metabolomics combined with machine learning for the identification of SARS-CoV-2 and other viruses directly from upper respiratory samples]() paper. It includes an `environment.yml` file to help you build the virtual environment for the software stack used to perform analysis.

## Requirements

This repository has a `requirements.txt` file which can be used in a virtual environment. [A good guide](https://www.freecodecamp.org/news/python-requirementstxt-explained/). 

The current version of this pipeline hosts:

1. Neural Net 1
2. Neural Net 2
3. Neural Net 3
4. LSTM RNN
5. SVM

It can also take any number of classes.

## Usage

```bash
python train_test.py -o <output_dir> -d <data_dir> -t <train_size>
```

or

```bash
bash run_multi.sh
```

**Note**: you will need to point the path to your Metabolomics multiclass dataset.

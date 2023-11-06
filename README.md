## Hokimiyon Muhammadjon, m.hokimiyon@innopolis.university, 	B21-AI-01

# Text De-toxification

## Overview

Text detoxification is the process of cleaning and improving the quality of text data, which can be useful for tasks like text analysis, sentiment analysis, and language modeling.

In this repository implemented 3 solution from scratch:
- Seq2Seq
- Attention
- Transformer

More detailed information written in `report` section, and all codes in section `src/models`. 

## Data

We used a dataset published in this [repository](https://github.com/s-nlp/detox) for training and testing our model. The data was preprocessed to remove noise and ensure data quality.


# How to use
## How to install
```bash
git clone https://github.com/WuBullMe/Text-De-Toxification.git

cd Text-De-Toxification
pip install -r requirements.txt
```

Install all requirements for the model, to avoid any errors. It's recommented to create a new python env before installing all those packages.

# How to Predict
## First go to models and download what model you want
```bash
cd models
```

# After download go back and do predict

```bash
cd ..
python src/models/predict_model.py --sentence "your sentence" --model "what model you downloaded"
```
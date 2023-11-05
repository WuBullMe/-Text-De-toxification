## Hokimiyon Muhammadjon, m.hokimiyon@innopolis.university, 	B21-AI-01

# Text De-toxification

Detoxification is an automatic transformation of a text such that:
- text becomes non-toxic
- the content of the text stays the same.

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
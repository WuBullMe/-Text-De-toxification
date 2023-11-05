import torch

from src.data.make_dataset import prepareData, get_dataloader, tensorFromSentence
from src.models.train_model import train

def get_mod(model_path):
    tp = 1
    if model_path.startswith("transformer"):
        from src.models.transformer import Transformer, Encoder, Decoder
        model = torch.load("models/transformer.pt")
    if model_path.startswith("attention"):
        from src.models.attention import Attention, Encoder, Decoder
        model = torch.load("models/attention.pt")
        tp = 0
    if model_path.startswith("seq2seq"):
        from src.models.seq2seq import Seq2Seq, Encoder, Decoder
        model = torch.load("models/seq2seq.pt")
        tp = 0
    
    return model, tp
import torch
import torch.nn as nn
import nltk
import string
import contractions
from spellchecker import SpellChecker

from seq2seq import get_seq2seq
from attention import get_attention
from transformer import get_transformer

import argparse
nltk.download('punkt')


# Convert every word in sentence to indexes of vocabulary
def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in nltk.word_tokenize(sentence.lower())]

# Convert every word in sentence to indexes of vocabulary in tensor format
def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def preprocess(sentence):
    spell = SpellChecker(distance=1)
    sentence = contractions.fix(sentence.lower())
    sentence = nltk.word_tokenize(sentence)
    words = []
    for word in sentence:
        for sub_word in word.split('-'):
            words.append(sub_word)
    new_row = []
    for word in words:
        res = spell.correction(word)
        if res is not None:
            new_row.append(res)
        else:
            if len([ch for ch in string.punctuation if ch in word]) == 0:
                new_row.append(word)
    sentence = ' '.join(new_row)
    return sentence

def evaluate(model, sentence, vocab_tox, vocab_detox):
    with torch.no_grad():
        model.eval()
        input_tensor = tensorFromSentence(vocab_tox, preprocess(sentence))

        outputs = model(input_tensor)

        _, topi = outputs.topk(1)
        ids = topi.squeeze()

        words = []
        for idx in ids:
            if idx.item() == EOS_token:
                break
            words.append(vocab_detox.index2word[idx.item()])
    return words

def evaluateSentence(model, vocab_tox, vocab_detox, sentence):
    print('origin:     ', sentence)
    output_words = evaluate(model, sentence, vocab_tox, vocab_detox)
    output_sentence = "".join([" "+i if not i.startswith("'") and not i.startswith("n'") and i not in string.punctuation else i for i in output_words]).strip()
    print('predicted:  ', output_sentence)
    print('')
        
# Accept model, and the sentence to be detoxified
def predict(model_path, sentence):
    if model_path == "transformer":
        model = get_transformer(model_path)
    if model_path == "attention":
        model = get_transformer(model_path)
    if model_path == "seq2seq":
        model = get_transformer(model_path)
    
    
    model.eval()
    evaluateSentence(model, model.encoder.vocab, model.decoder.vocab, sentence)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--sentence', type=str)
    p.add_argument('--model', type=str)
    args = p.parse_args()
    print(predict(args.model, args.sentence))
    
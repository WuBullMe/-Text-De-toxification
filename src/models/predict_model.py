import torch
import torch.nn as nn
import nltk
nltk.download('punkt')


# Convert every word in sentence to indexes of vocabulary
def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in nltk.word_tokenize(sentence.lower())]

# Convert every word in sentence to indexes of vocabulary in tensor format
def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def evaluate(model, sentence, vocab_tox, vocab_detox):
    with torch.no_grad():
        model.eval()
        input_tensor = tensorFromSentence(vocab_tox, sentence)

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
        

def predict(model, sentence):
    
    trained_model = torch.load(model)
    trained_model.eval()
    evaluateSentence(trained_model, trained_model.encoder.vocab, trained_model.decoder.vocab, sentence)
    
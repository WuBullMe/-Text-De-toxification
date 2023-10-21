import torch
import torch.nn as nn

class Attention(nn.Module):
    """
        Attention model
    """
    def __init__(self, encoder, decoder):
        """
            :param input_size:   number of possible initial words
            :param hidden_size:  number of features in the hidden state
            :param output_size:  number of possible result words
            :param dropout_p:    probability for dropout
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        
    def forward(self, input, target=None):
        encoder_outputs, encoder_hidden = self.encoder(input)
        decoder_outputs = self.decoder(encoder_outputs, encoder_hidden, target)
        return decoder_outputs
        

class Encoder(nn.Module):
    """
        Encoder part
    """
    def __init__(self, input_size, embed_size, hidden_size, vocab, device="cpu", max_length=0, dropout_p=0.1):
        """
            :param input_size:  number of possible initial words
            :param hidden_size: number of features in the hidden state
            :param dropout_p:   probability for dropout
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=2, dropout=dropout_p, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
    
class Decoder(nn.Module):
    """
        Decoder part
    """
    def __init__(self, embed_size, hidden_size, output_size, vocab, device="cpu", max_length = 0, dropout_p=0.1):
        """
            :param hidden_size:  number of features in the hidden state
            :param output_size:  number of possible result words
            :param dropout_p:    probability for dropout
        """
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=2, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.vocab.word2index['<sos>'])
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        
        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        
        return output, hidden

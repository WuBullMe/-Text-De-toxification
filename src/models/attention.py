import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """
        Attention head of heads
    """
    def __init__(self, embed_size, head_size, batch_size, mask=False, dropout=0.2):
        super().__init__()
        self.mask = mask
        
        self.query = nn.Linear(embed_size, head_size)
        self.key = nn.Linear(embed_size, head_size)
        self.value = nn.Linear(embed_size, head_size)
        if mask:
            self.register_buffer('tril', torch.tril(torch.ones(batch_size, batch_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, crossData=None):
        """
            :param crossData: 
                * crossData = None -> It's Self Attention
                * else             -> It's Cross Attention
                
                
            B -> batch size
            T -> series length
            C -> size of embedding
        """
        B,T,C = data.shape
        query = self.query(data)
        if crossData is not None:
            key = self.key(crossData)
            v = self.value(crossData)
        else:
            key = self.key(data)
            v = self.value(data)
        
        wei = query @ key.transpose(1, 2) * C**-0.5 # (B, 1, C) @ (B, C, T) -> (B, 1, T)
        if self.mask:
            wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v # (B, 1, T) @ (B, T, C) -> (B, 1, C)
        return out
        
        
class MultipleAttention(nn.Module):
    """
        Multiple head for Attention
    """
    def __init__(self, embed_size, n_head, head_size, batch_size, dropout=0.2):
        super().__init__()
        self.blockHead = nn.ModuleList([AttentionHead(embed_size, head_size, batch_size) for _ in range(n_head)])
        self.fc = nn.Linear(head_size * n_head, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data, crossData=None):
        heads = torch.cat([layer(data, crossData=crossData) for layer in self.blockHead], dim=-1)
        
        out = self.dropout(self.fc(heads))
        return out

        
class Attention(nn.Module):
    """
        Attention model
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        
    def forward(self, input, target=None, teacher_force=1.0):
        encoder_outputs, encoder_hidden = self.encoder(input)
        decoder_outputs = self.decoder(encoder_outputs, encoder_hidden, target, teacher_force)
        return decoder_outputs
        

class Encoder(nn.Module):
    """
        Encoder part
    """
    def __init__(self, input_size, batch_size, embed_size, hidden_size, vocab, device=torch.device("cpu"), max_length=0, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        embedded = self.embedding(data)
        out, hidden = self.gru(embedded)
        
        return out, hidden
    
    
class Decoder(nn.Module):
    """
        Decoder part
    """
    def __init__(self, n_head, batch_size, embed_size, hidden_size, output_size, vocab, device=torch.device("cpu"), max_length = 0, dropout=0.2):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=self.vocab.word2index['<pad>'])
        self.gru = nn.GRU(embed_size * 2, hidden_size, batch_first=True)
        self.blocks = MultipleAttention(embed_size, n_head, embed_size//n_head, batch_size, dropout)
        self.out = nn.Linear(hidden_size, output_size)

    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_force=1.0):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.vocab.word2index['<sos>'])
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        if target_tensor is not None:
            B = target_tensor.shape[0]
            target_tensor = torch.cat((target_tensor, torch.ones(B, 1, device=self.device, dtype=torch.int64)), dim=-1)
        for i in range(1, self.max_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            
            if target_tensor is not None and torch.rand(1) < teacher_force:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs

    def forward_step(self, data, hidden, encoder_outputs):
        embed = self.embedding(data)
        
        query = hidden.permute(1, 0, 2)
        attn = self.blocks(query, encoder_outputs)
        output = torch.cat((attn, embed), dim=-1)
        
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        
        return output, hidden

def get_attention():
    return torch.load("model/attention.pt")

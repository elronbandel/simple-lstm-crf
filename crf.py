from torch.nn import  Parameter, Module, Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn, Tensor
import torch


# this is compressed version of the modification of 'Allennlp' crf modified in https://github.com/yumoh/torchcrf
class CRF(Module):
    def __init__(self, num_tags, batch_first=False):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = Parameter(Tensor(num_tags).normal_(0, 0.1))
        self.end_transitions = Parameter(Tensor(num_tags).normal_(0, 0.1))
        self.transitions = Parameter(Tensor(num_tags, num_tags).normal_(0, 0.1))

    def forward(self, emissions, tags, mask, reduction_fn=torch.mean) -> torch.Tensor:
        mask = torch.ones_like(tags, dtype=torch.uint8) if mask is None else mask
        if self.batch_first:
            emissions, tags, mask = emissions.transpose(0, 1), tags.transpose(0, 1), mask.transpose(0, 1)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        #return nll loss
        return reduction_fn(denominator - numerator)

    def decode(self, emissions, mask=None):
        mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8) if mask is None else mask
        if self.batch_first:
            emissions, mask = emissions.transpose(0, 1), mask.transpose(0, 1)
        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        seq_length, batch_size = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions, mask):
        seq_len = emissions.size(0)
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask):
        seq_len, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class LSTMCRF(Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, bidirectional=True, num_layers=2, device=None, dropout=0):
        super().__init__()
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        lstm_hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.seq_pad_idx, self.tag_pad_idx = vocab_size, num_tags + 1
        self.embedding = Embedding(vocab_size + 1, embed_dim, padding_idx=self.seq_pad_idx)
        self.lstm = LSTM(embed_dim, lstm_hidden_dim, bidirectional=bidirectional, num_layers=num_layers
                         , batch_first=True, dropout=dropout)
        self.lstm2crf = Linear(hidden_dim, num_tags + 2)
        self.crf = CRF(num_tags + 2, batch_first=True).to(self.device)

    def forward(self, seqs, tags):
        lens = list(map(len, seqs))
        tags = pad_sequence(tags, batch_first=True, padding_value=self.tag_pad_idx).to(self.device)
        seqs = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_idx).to(self.device)
        embeded = self.embedding(seqs)
        packed = pack_padded_sequence(embeded, lens, batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(packed)
        lstm_out, out_lens = pad_packed_sequence(lstm_out, batch_first=True)
        emissions = self.lstm2crf(lstm_out)
        if self.training:
            crf_nll_loss = self.crf(emissions, tags, mask=tags.lt(self.tag_pad_idx))
            return crf_nll_loss
        else:
            crf_opt_tags =  self.crf.decode(emissions, mask=tags.lt(self.tag_pad_idx))
            return crf_opt_tags

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from vocab import Vocab
import utils


class BiLSTMCRF(nn.Module):
    def __init__(self, num_classes, feature_size, dropout_rate=0.5, hidden_size=256):
        """ Initialize the model
        Args:
            num_classes:
            feature_size (int): embedding size, # of features
            hidden_size (int): hidden state size

            len: how many discrete inputs come from the data point (number of words)
            b: batch size
            e: # of features per input (word)
            K: # of classes (I think)
        """
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.feature_size = feature_size
        self.hidden_size = hidden_size  # "the number of features in the hidden state" of the LSTM
        self.num_classes = num_classes  # labels
        self.dropout = nn.Dropout(dropout_rate)

        self.LSTM = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, bidirectional=True)

        # a linear layer from hidden size * 2 to # of possible labels? or # of samples?
        # I think len(self.tag_vocab) = K, meaning K is the number of classes
        # emit_score is emission score
        self.hidden2emit_score = nn.Linear(hidden_size * 2, self.num_classes)

        # CRF transition matrix, shape: (K, K)
        self.transition = nn.Parameter(torch.randn(self.num_classes, self.num_classes))

    def forward(self, input_batch, labels):
        """
        1. takes in sentences, featurizes them (embeds them in feature space)
        2. calls encode on them, which puts them through the BiLSTM
        3. calculates and returns the CRF loss

        Args:
            input_batch (tensor): sentences, shape (b, len, e). len is the (max?) number of segments
            labels (tensor): corresponding tags, shape (b, len)
            sen_lengths (list): sentence lengths
        Returns:
            loss (tensor): loss on the batch, shape (b,)
        """
        # mask has the same size as input_batch
        # it has True in every place where the word is not <PAD> and false where the word is <PAD>
        # mask = (input_batch != self.sent_vocab[self.sent_vocab.PAD]).to(self.device)  # shape: (b, len)
        # for now, let's have mask be an entire array of True
        # later we can try removing mask entirely, since I don't think it is necessary

        # input shape (b, s, N_MELS, t)
        mask = torch.full(input_batch.shape, True, dtype=bool)
        input_batch = input_batch.transpose(0, 1)  # shape: (len, b, e)
        emit_score = self.encode(input_batch)  # shape: (b, len, K)
        loss = self.cal_loss(labels, mask, emit_score)  # shape: (b,)
        return loss

    def encode(self, input_batch):
        """ BiLSTM Encoder

        puts featurized sentences thru the BiLSTM
        not sure what emit_score is or what hidden2emit_score does

        Args:
            input_batch (tensor): sentences with word embeddings, shape (len, b, e)
            sent_lengths (list): sentence lengths
        Returns:
            emit_score (tensor): emit score, shape (b, len, K)
        """
        # I think we won't need to do any padding, so can probably remove sent_lengths param
        # padded_sentences = pack_padded_sequence(input_batch, sent_lengths)
        hidden_states, _ = self.LSTM(input_batch)
        # hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)  # shape: (b, len, 2h)
        emit_score = self.hidden2emit_score(hidden_states)  # shape: (b, len, K)
        emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return emit_score

    def cal_loss(self, labels, mask, emit_score):
        """ Calculate CRF loss

        uses the CRF transition matrix

        Args:
            labels (tensor): a batch of tags, shape (b, len)
            mask (tensor): mask for the tags, shape (b, len),
                values in PAD position is 0, otherwise 1
            emit_score (tensor): emit matrix, shape (b, len, K)
        Returns:
            loss (tensor): loss of the batch, shape (b,)
        """
        batch_size, num_classes = labels.shape
        # calculate score for the labels
        index = labels.unsqueeze(dim=2).long()  # had to cast this from float to long in order for the next line to run
        score = torch.gather(emit_score, dim=2, index=index)
        score = score.squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += self.transition[labels[:, :-1], labels[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, num_classes):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # shape: (b,)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)
        return loss

    def predict(self, sentences, sen_lengths):
        """
        makes label predictions for input sentences
        embeds sentences in feature space
        encodes embedded sentences
        uses transition matrix

        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            sen_lengths (list): sentence lengths
        Returns:
            tags (list[list[str]]): predicted tags for the batch
        """
        batch_size = sentences.shape[0]
        # mask = (sentences != self.sent_vocab[self.sent_vocab.PAD])  # shape: (b, len)
        mask = torch.full(sentences.shape, True, dtype=bool)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        emit_score = self.encode(sentences, sen_lengths)  # shape: (b, len, K)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = {
            'num_classes': self.num_classes,
            'args': dict(dropout_rate=self.dropout_rate, feature_size=self.feature_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(params['sent_vocab'], params['num_classes'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device


def main():
    sent_vocab = Vocab.load('./vocab/sent_vocab.json')
    tag_vocab = Vocab.load('./vocab/tag_vocab.json')
    train_data, dev_data = utils.generate_train_dev_dataset('./data/train.txt', sent_vocab, tag_vocab)
    device = torch.device('cpu')
    model = BiLSTMCRF(sent_vocab, tag_vocab)
    model.to(device)
    model.save('./model/model.pth')
    model = model.load('./model/model.pth', device)


if __name__ == '__main__':
    main()
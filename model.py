import torch
from torch import nn
from transformers import BertModel, RobertaModel
import geoopt as gt
import math
class NGramers(nn.Module):
    def __init__(self, input_size, hidden_size, max_gram, dropout_rate):
        super().__init__()

        self.cnn_list = nn.ModuleList([nn.Conv1d(in_channels=input_size,
                                                 out_channels=hidden_size,
                                                 kernel_size=n) for n in range(1, max_gram + 1)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.transpose(1, 2)

        cnn_outpus = []
        for cnn in self.cnn_list:
            y = cnn(x)
            y = self.relu(y)
            y = self.dropout(y)
            cnn_outpus.append(y.transpose(1, 2))
        outputs = torch.cat(cnn_outpus, dim=1)
        return outputs

class MatchSum(nn.Module):

    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()

        self.hidden_size = hidden_size
        self.candidate_num = candidate_num

        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained('../MatchSum/transformers_model/bert-base-uncased',
                                                     output_hidden_states=True)
        else:
            self.encoder = RobertaModel.from_pretrained('../MatchSum/transformers_model/roberta-base',
                                                        output_hidden_states=True)
        self.ball = gt.PoincareBall(0.9)
        self.rank = 512
        self.trans_d = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.trans_s = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.trans_c = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.trans_p = nn.Linear(self.rank, self.rank, bias=False)
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.cnn2gram = NGramers(input_size=self.hidden_size,
                                    hidden_size=self.rank,
                                    max_gram=2,
                                    dropout_rate=0.1)
        self.mse = torch.nn.MSELoss(reduction="mean")

    def forward(self, text_id, candidate_id, summary_id):
        batch_size = text_id.size(0)
        pad_id = 0  # for BERT
        if text_id[0][0] == 0:
            pad_id = 1  # for RoBERTa

        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0]

        doc_emb = self.ball.expmap0(self.trans_d(out[:, 0, :]))
        doc_word_emb = self.ball.expmap0(self.trans_p(self.cnn2gram(out[:, 1:-1, :])))

        doc_length = doc_word_emb.shape[1]
        document_interactions = 1/self.ball.dist(doc_word_emb, doc_emb.unsqueeze(1).repeat(1, doc_length, 1))
        
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0]  # last layer

        e_summary_emb,_ = torch.max(out, dim=1)
        e_summary_emb = self.trans_s(e_summary_emb)

        summary_emb = self.ball.expmap0(e_summary_emb)
        summary_score = -self.ball.dist2(summary_emb, doc_emb)

        summary_interactions = 1/self.ball.dist(doc_word_emb, summary_emb.unsqueeze(1).repeat(1, doc_length, 1))
        summary_interaction_score = torch.cosine_similarity(document_interactions, summary_interactions, dim=-1)
        summary_score = summary_score + summary_interaction_score

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]

        e_candidate_emb,_ = torch.max(out, dim=1)
        e_candidate_emb = self.trans_c(e_candidate_emb).view(batch_size, candidate_num, self.rank)

        candidate_emb = self.ball.expmap0(e_candidate_emb)  # [batch_size, candidate_num, hidden_size]
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = -self.ball.dist2(candidate_emb, doc_emb) # [batch_size, candidate_num]
        candidate_interactions = 1/self.ball.dist(doc_word_emb.unsqueeze(1).repeat(1, candidate_emb.shape[1], 1, 1), candidate_emb.unsqueeze(2).repeat(1, 1, doc_word_emb.shape[1], 1))
        candidate_interaction_score = torch.cosine_similarity(document_interactions.unsqueeze(1).repeat(1, candidate_emb.shape[1], 1), candidate_interactions, dim=-1)

        score = score + candidate_interaction_score

        return {'score': score, 'summary_score': summary_score}

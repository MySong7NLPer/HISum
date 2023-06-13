import torch
from torch import nn
from torch.nn import init
from hyperrnn import HyperGRU
from transformers import BertModel, RobertaModel
import geoopt as gt
import math


class NGramers(nn.Module):
    def __init__(self, input_size, hidden_size, max_gram, dropout_rate):
        super().__init__()

        self.cnn_list = nn.ModuleList([nn.Conv1d(in_channels=input_size,
                                                 out_channels=hidden_size,
                                                 kernel_size=n, ) for n in range(1, max_gram + 1)])
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
        # outputs = torch.cat(cnn_outpus, dim=1)

        return cnn_outpus


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

        self.ball = gt.PoincareBall()
        # self.max_pooling = nn.AdaptiveMaxPool1d(512)
        # self.avg_pooling = nn.AdaptiveAvgPool1d(512)
        self.rank = 768
        # self.trans = nn.Linear(768, self.rank)
        # self.trans_can = nn.Linear(768, 128)
        # self.trans_summary = nn.Linear(768, 128)

        self.cls = nn.Linear(6, 1)
        # self.final = nn.Linear(256, 1, bias=False)
        # self.dropout = nn.Dropout(0.1)

        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

        max_gram = 3
        cnn_output_size = 768

        self.cnn2gram = NGramers(input_size=hidden_size,
                                 hidden_size=cnn_output_size,
                                 max_gram=max_gram,
                                 dropout_rate=0.1).cuda()

    def forward(self, text_id, candidate_id, summary_id):

        batch_size = text_id.size(0)

        pad_id = 0  # for BERT
        if text_id[0][0] == 0:
            pad_id = 1  # for RoBERTa

        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0]
        # last layer
        # doc_emb = self.ball.expmap0(out[:, 1:-1, :])
        doc_emb = out[:, 1:-1, :]
        doc_emb_multiple = self.cnn2gram(doc_emb)
        # [batch_size, 256, hidden_size]
        # doc_emb_whole = self.ball.expmap0(self.trans(out[:, 0, :])).unsqueeze(1)
        # [batch_size, 1, hidden_size]
        # doc_emb_sim = -self.ball.dist2(doc_emb, doc_emb_whole.expand_as(doc_emb)).unsqueeze(1) / math.sqrt(self.rank)
        # doc_emb_sim_max = self.mean_max_pooling(doc_emb_sim)
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0]  # last layer
        summary_emb = self.ball.expmap0(out[:, 1:-1, :])
        summary_emb = self.einstein_midpoint(summary_emb)

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        candidate_emb = self.ball.expmap0(self.encoder(candidate_id, attention_mask=input_mask)[0][:, 1:-1, :])
        candidate_emb = self.einstein_midpoint(candidate_emb)
        candidate_emb = candidate_emb.view(batch_size, candidate_num,
                                           self.rank)  # [batch_size, candidate_num, hidden_size]

        uni_interaction_summary, uni_interaction_score = self.uni_segement(candidate_emb, summary_emb, candidate_num,
                                                                           doc_emb_multiple[0])
        bi_interaction_summary, bi_interaction_score = self.uni_segement(candidate_emb, summary_emb, candidate_num,
                                                                         doc_emb_multiple[1])
        tri_interaction_summary, tri_interaction_score = self.uni_segement(candidate_emb, summary_emb, candidate_num,
                                                                           doc_emb_multiple[2])

        summary_score = self.cls(
            torch.cat([uni_interaction_summary, bi_interaction_summary, tri_interaction_summary], -1))
        score = self.cls(torch.cat([uni_interaction_score, bi_interaction_score, tri_interaction_score], -1))

        import pdb;
        pdb.set_trace()
        return {'score': score, 'summary_score': summary_score}

    def uni_segement(self, candidate_emb, summary_emb, candidate_num, doc_emb_multiple):

        doc_emb = doc_emb_multiple
        # get summary embedding

        summary_emb = summary_emb.unsqueeze(1).repeat(1, doc_emb.shape[1], 1)  # [batch_size, hidden_size]
        # get summary score

        tmp_summary_score = -self.ball.dist2(doc_emb, summary_emb).unsqueeze(1) / math.sqrt(self.rank)
        tmp_summary_score_max = self.mean_max_pooling(tmp_summary_score)
        # tmp_summary_score_mean = self.avg_pooling(tmp_summary_score)
        # tmp_summary_score = torch.cat((tmp_summary_score_max, tmp_summary_score_mean), -1).squeeze(1)
        # summary_score_max, _ = torch.max(tmp_summary_score, -1)
        # summary_score_mean = torch.mean(tmp_summary_score, -1)
        # summary_score = summary_score_max + summary_score_mean
        # summary_score = torch.sigmoid(self.final(torch.tanh(self.cls(tmp_summary_score_max)))).squeeze(-1)
        # summary_score = torch.cosine_similarity(doc_emb_sim_max, tmp_summary_score_max, dim=-1)
        # summary_score = summary_score.view(batch_size)
        # summary_score = torch.mean(tmp_summary_score_max, dim=-1).view(batch_size)
        summary_score = tmp_summary_score_max

        candidate_emb = candidate_emb.unsqueeze(2).repeat(1, 1, doc_emb.shape[1], 1)
        doc_emb = doc_emb.unsqueeze(1).repeat(1, candidate_num, 1, 1)

        # assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)

        # get candidate score
        # doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        tmp_score = -self.ball.dist2(candidate_emb, doc_emb) / math.sqrt(self.rank)  # [batch_size, candidate_num]
        # score = torch.mean(self.mean_max_pooling(tmp_score), -1)
        score = self.mean_max_pooling(tmp_score)
        # tmp_score_mean = self.avg_pooling(tmp_score)
        # tmp_score = torch.cat((tmp_score_max, tmp_score_mean), -1)
        # score_max, _ = torch.max(tmp_score, -1)
        # score_mean = torch.mean(tmp_score, -1)
        # score = score_max + score_mean
        # score = torch.sigmoid(self.final(torch.tanh(self.cls(tmp_score_max))).view(batch_size, candidate_num))
        # import pdb; pdb.set_trace()
        # print(score)
        # score = torch.cosine_similarity(doc_emb_sim_max.expand_as(tmp_score_max), tmp_score_max, dim=-1).view(batch_size, candidate_num)

        # assert score.size() == (batch_size, candidate_num)

        return summary_score, score

    def mean_max_pooling(self, relevance):
        # emb = [batch, doc_length, doc_length]
        max_signals, _ = torch.max(relevance, -1)
        mean_signals = torch.mean(relevance, -1).unsqueeze(-1)
        # mean_signals = [batch, doc_length, 1]
        return torch.cat([max_signals.unsqueeze(-1), mean_signals], -1)  # [batch, doc_length, 2]

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def klein_constraint(self, x):
        last_dim_val = x.size(-1)
        norm = torch.reshape(torch.norm(x, dim=-1), [-1, 1])
        maxnorm = (1 - self.eps[x.dtype])
        cond = norm > maxnorm
        x_reshape = torch.reshape(x, [-1, last_dim_val])
        projected = x_reshape / (norm + self.min_norm) * maxnorm
        x_reshape = torch.where(cond, projected, x_reshape)
        x = torch.reshape(x_reshape, list(x.size()))
        return x

    def to_klein(self, x, c=1):
        x_2 = torch.sum(x * x, dim=-1, keepdim=True)
        x_klein = 2 * x / (1.0 + x_2)
        x_klein = self.klein_constraint(x_klein)
        return x_klein

    def klein_to_poincare(self, x, c=1):
        x_poincare = x / (1.0 + torch.sqrt(1.0 - torch.sum(x * x, dim=-1, keepdim=True)))
        x_poincare = self.proj(x_poincare, c)
        return x_poincare

    def lorentz_factors(self, x):
        x_norm = torch.norm(x, dim=-1)
        return 1.0 / (1.0 - x_norm ** 2 + self.min_norm)

    def einstein_midpoint(self, x, c=1):
        x = self.to_klein(x, c)
        x_lorentz = self.lorentz_factors(x)
        x_norm = torch.norm(x, dim=-1)
        # deal with pad value
        x_lorentz = (1.0 - torch._cast_Float(x_norm == 0.0)) * x_lorentz
        x_lorentz_sum = torch.sum(x_lorentz, dim=-1, keepdim=True)
        x_lorentz_expand = torch.unsqueeze(x_lorentz, dim=-1)
        x_midpoint = torch.sum(x_lorentz_expand * x, dim=1) / x_lorentz_sum
        x_midpoint = self.klein_constraint(x_midpoint)
        x_p = self.klein_to_poincare(x_midpoint, c)
        return x_p
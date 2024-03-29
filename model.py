import torch
import numpy as np
from torch.nn import *
from transformers import AutoModel
import torch.nn.functional as F
from layers import GraphConvolution

class BertEmbedding(Module):

    def __init__(self, args):
        super(BertEmbedding, self).__init__()
        self.device = args.device
        print('Init Bert')
        self.bert = AutoModel.from_pretrained(args.bert_type)

        if not args.update_bert:
            for params in self.bert.parameters():
                params.requires_grad = False
        self.output_size = 8 * self.bert.config.hidden_size

    def forward(self, inputs):

        BL = inputs['bert_longest_length']
        WL = inputs['word_longest_length']

        indices = inputs['indices'][:, :BL].to(self.device)
        # print(indices.shape)
        attention_mask = inputs['attention_mask'][:, :BL].to(self.device)

        bert_outputs = self.bert(indices,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)

        bert_x = torch.concat(bert_outputs['hidden_states'][-8:], dim=-1)  # B x L x D
        # print('Bert x', tuple(bert_x.shape))
        # print(inputs['bert_length'])
        # print(inputs['word_length'])
        # print(bert_x.shape)
        transforms = inputs['transform_matrix'][:, :WL, :BL].to(self.device)
        embeddings = torch.bmm(transforms, bert_x)

        return embeddings

class unsqueeze(Module):
    def __init__(self, idx):
        super(transpose, self).__init__()
        self.idx = idx

    def forward(self, x):
        return torch.unsqueeze(x, self.idx)

class transpose(Module):
    def __init__(self):
        super(transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)

class padding(Module):
    def __init__(self, num_pad=3):
        super(padding, self).__init__()
        self.num_pad = num_pad

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        pad = x[0].unsqueeze(0)
        for i in range(1, self.num_pad):
            x = torch.cat((pad, x), 0)
        return torch.transpose(x, 0, 1)

class SelectItem(Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Print(Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class CNNModel(Module):

    def __init__(self, args):
        super(CNNModel, self).__init__()
        self.device = args.device
        self.c = c = len(args.label2index)

        self.Ci = Ci = 8 * 768
        self.embeddings = BertEmbedding(args)

        self.fc = Sequential(
            Conv1d(Ci, 1024, padding='same', kernel_size=3),
            ReLU(),
            Dropout(0.2),
            Conv1d(1024, 1024, padding='same', kernel_size=3),
            ReLU(),
            Dropout(0.2),
            transpose(),
            Linear(1024, 256),
            ReLU(),
            Dropout(),
            Linear(256, self.c),
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)  # B x L x D
        x = x.transpose(1, 2)
        logits = self.fc(x)
        logits = logits.view(-1, logits.shape[-1])
        preds = torch.argmax(logits, dim=-1)
        return logits, preds

class LSTMModel(Module):

    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.device = args.device
        self.c = c = len(args.label2index)

        self.Ci = Ci = 8 * 768
        self.embeddings = BertEmbedding(args)

        self.fc = Sequential(
            LSTM(Ci, 1024, 2, dropout=0.3),
            SelectItem(0),
            ReLU(),
            Linear(1024, 512),
            Dropout(0.3),
            ReLU(),
            Linear(512, self.c),

        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.view(-1, x.shape[-1])
        logits = self.fc(x)
        preds = torch.argmax(logits, dim=-1)
        return logits, preds

class GRUModel(Module):

    def __init__(self, args):
        super(GRUModel, self).__init__()
        self.device = args.device
        self.c = c = len(args.label2index)

        self.Ci = Ci = 8 * 768
        self.embeddings = BertEmbedding(args)

        self.fc = Sequential(
            GRU(8*768, 1024, dropout=0.3),
            SelectItem(0),
            ReLU(),
            Linear(1024, 1024),
            Dropout(0.3),
            ReLU(),
            Linear(1024, self.c),

        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.view(-1, x.shape[-1])
        logits = self.fc(x)
        preds = torch.argmax(logits, dim=-1)
        return logits, preds


def diagonal_block(adj_matrix1, adj_matrix2):
    adj_shape1 = adj_matrix1.shape
    adj_shape2 = adj_matrix2.shape
    Z1 = np.zeros((adj_shape1[0], adj_shape2[1]), dtype=np.float32)
    Z2 = np.zeros((adj_shape2[1], adj_shape1[0]), dtype=np.float32)
    adj_matrix = np.asarray(np.bmat([[adj_matrix1, Z1], [Z2, adj_matrix2]]))
    return adj_matrix

class GCNModel(Module):
    def __init__(self, args):
        super(GCNModel, self).__init__()
        self.device = args.device
        self.c = c = len(args.label2index)

        self.Ci = Ci = 8 * 768
        self.embeddings = BertEmbedding(args)

        self.gc_dim = 256
        self.lstm = LSTM(Ci, self.gc_dim, bidirectional=True, batch_first=True, num_layers=1)
        self.prj = Linear(768 * 8, self.gc_dim)
        self.gc1 = GraphConvolution(2 * self.gc_dim, 2 * self.gc_dim)
        self.gc2 = GraphConvolution(2 * self.gc_dim, 2 * self.gc_dim)
        self.all_dim = 768 * 8 + self.gc_dim * 6

        self.fc = Sequential(
            Linear(self.all_dim, 1024),
            Dropout(),
            ReLU(),
            Linear(1024, self.c)
        )
    def forward(self, inputs):

        embeddings = self.embeddings(inputs)

        adj_matrix = inputs['adj_matrix'].to(self.device)
        adj_leng = embeddings.shape[1]
        adj_matrix = adj_matrix[:, :adj_leng, :adj_leng]

        x, _ = self.lstm(embeddings)

        output1 = self.gc1(F.relu(x), adj_matrix)  # L, D
        output2 = self.gc2(F.relu(output1), adj_matrix)  # L, D
        final_reps = torch.cat([x, embeddings, output1, output2], dim=-1)  # L, xxxD
        final_reps = final_reps.view(-1, final_reps.shape[-1])
        logits = self.fc(final_reps)  # B*L, C
        preds = torch.argmax(logits, dim=-1)
        return logits, preds

if __name__ == '__main__':
    pass
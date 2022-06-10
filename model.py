import torch
import numpy as np
from torch.nn import *
from transformers import AutoModel


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

        BL = torch.max(inputs['bert_length'])

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
        all_embeddings = []
        for sid, sentence_spans in enumerate(inputs['word_spans']):
            for start, end in sentence_spans:
                emb = torch.mean(bert_x[sid][start:end], dim=-2)
                # print('emb', tuple(emb.shape))

                all_embeddings.append(emb)

        all_embeddings = torch.stack(all_embeddings)

        return all_embeddings


class transpose(Module):
    def __init__(self):
        super(transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 0, 1)

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


class MLPModel(Module):

    def __init__(self, args):
        super(MLPModel, self).__init__()
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
        x = torch.transpose(x, 0, 1)
        logits = self.fc(x)
        preds = torch.argmax(logits, dim=-1)
        return logits, preds

if __name__ == '__main__':
    pass
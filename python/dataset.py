import torch
from torch.nn.utils.rnn import pad_sequence

def prepare_dataset(sequences,char2idx):
    X,Y = [],[]
    for seq in sequences:
        ids = [char2idx[c] for c in seq]
        if len(ids) < 2 :continue
        X.append(torch.tensor(ids[:-1]))
        Y.append(torch.tensor(ids[1:]))
    return pad_sequence(X,batch_first=True),pad_sequence(Y,batch_first=True)

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class CNNRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = 512
        hidden_dim = 128
        num_layers = 1
        self.cnn = models.resnet50(pretrained=True)
        out_features = self.cnn.fc.in_features
        self.fc1 = nn.Linear(out_features, input_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, vid, lengths):

        B, T, *a = vid.shape
        vid = vid.permute(0, 1, 4, 2, 3)
        outs = []

        def hook(module, input, output):
            outs.append(input)

        self.cnn.fc.register_forward_hook(hook)
        for t in range(T):
            # print(t)
            frame = vid[:, t, :, :, :]
            out = self.cnn(frame)

        if outs[0][0].ndim == 2:
            outs = [ten[0].unsqueeze(0) for ten in outs]
        else:
            outs = [ten[0] for ten in outs]
        outs = torch.cat(outs, dim=1)
        outs = self.fc1(outs)
        packed_seq = pack_padded_sequence(outs, lengths, batch_first=True, enforce_sorted=False)

        out, hn = self.rnn(packed_seq)
        padded_seq, lengths = pad_packed_sequence(out, batch_first=True)
        out = self.fc2(padded_seq)
        return out

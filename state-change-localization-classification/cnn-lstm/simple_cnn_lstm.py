import torch
import torch.nn as nn
import torchvision
import types

def forward_reimpl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x.squeeze(2).squeeze(2)


class cnnlstm(nn.Module):
    def __init__(self, hidden_size=512, num_layers=1, state=False):
        super(cnnlstm, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = None
        self.lstm = nn.LSTM(2048, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.regressor = nn.Linear(hidden_size*2, 1)
        self.state = state
        if self.state:
            self.state_classifier = nn.Linear(hidden_size*2, 2)
        self.backbone.forward = types.MethodType(forward_reimpl, self.backbone)
        
    def forward(self, x):
        #x: (b, c, seq_len, h, w)
        seq_len = x.shape[2]
        batch_size = x.shape[0]
        x = x.permute((0,2,1,3,4))
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.backbone(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)  # (b, seq_len, hidden_size*2)
        out = self.regressor(x).squeeze(2)
        if self.state:
            state = self.state_classifier(x.mean(1))
            return torch.sigmoid(out), state
        return torch.sigmoid(out)


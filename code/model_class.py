from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout, Tanh
import torch
import torch.nn as nn

DATASET = 'davis'
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(
            Conv1d(1, 32, kernel_size=4, stride=2, padding=0),
            BatchNorm1d(32),
            ReLU(inplace=True),
            Conv1d(32, 64, kernel_size=8, stride=2, padding=0),
            ReLU(inplace=True),
            MaxPool1d(3, stride=2),

        )

    def forward(self, x):
        x = x.float()
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return x


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.cnn_layers = Sequential(
            Conv1d(1, 32, kernel_size=4, stride=2, padding=0),
            BatchNorm1d(32),
            ReLU(inplace=True),
            Conv1d(32, 64, kernel_size=8, stride=2, padding=0),
            BatchNorm1d(64),
            ReLU(inplace=True),
            Conv1d(64, 96, kernel_size=12, stride=2, padding=0),
            ReLU(inplace=True),
            MaxPool1d(4, stride=3),

        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return x


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_dim = 1
        self.hidden_dim = 256
        self.target_dim = 1
        self.batch_size = 1
        self.p_len = 450
        self.s_len = 95

        if DATASET == "davis":
            self.input_n = 576
        if DATASET == "kiba":
            self.input_n = 640

        self.cnn = CNN()
        self.cnn2 = CNN2()
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=1)
        # 384 448
        self.linear1 = nn.Linear(self.input_n + self.hidden_dim, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = Dropout(0.1)

    def forward(self, x1, x2):

        h0 = torch.zeros(1, self.batch_size, self.hidden_dim).requires_grad_()
        # h0 = h0.to(device)
        c0 = torch.zeros(1, self.batch_size, self.hidden_dim).requires_grad_()
        # c0 = c0.to(device)

        x1 = x1.float()
        x2 = x2.float()

        c_in = x2.view(self.batch_size, 1, self.s_len)
        # print(c_in.shape)
        SMILEs = self.cnn(c_in)

        c_in_2 = x1.view(self.batch_size, 1, self.p_len)
        pre_proteins = self.cnn2(c_in_2)
        lstm_in = pre_proteins.view(len(x1), 1536, -1)
        proteins, (hn, cn) = self.lstm(lstm_in, (h0.detach(), c0.detach()))

        proteins = proteins[:, -1, :]
        concatination = torch.cat((SMILEs, proteins), 1)
        out1 = self.linear1(concatination)
        relu_out1 = self.relu(out1)
        relu_out1 = self.dropout(relu_out1)
        out2 = self.linear2(relu_out1)
        relu_out2 = self.relu(out2)
        relu_out2 = self.dropout(relu_out2)
        y_pred = self.linear3(relu_out2)

        return y_pred

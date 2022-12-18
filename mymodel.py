import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torchvision.models as models


##############################
#           LSTM
##############################
class LSTM(nn.Module):
    def __init__(self, num_classes, latent_dim, num_layers, hidden_dim, clips_num, bidirectional, device):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.final = nn.Sequential(
            nn.Linear(2 * hidden_dim * clips_num if bidirectional else hidden_dim * clips_num, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
            # nn.Softmax(dim=-1)
        )
        self.hidden_state = None
        self.device = device
        self.num_layers = num_layers
        self.directions_count = 2 if bidirectional else 1
        self.rnn_hidden_dim = hidden_dim


    def init_hidden(self, input_batch_size):
        """
        input_batch_size: the input batch which is different to the batch_size claimed in the dataset. This input_batch_size is
        used to initialize the size of the hidden_state (num_layer*direction, input_batch_size, hidden_size)

        The function outputs the zeros tensor, which is the leaf node
        https://discuss.pytorch.org/t/how-to-handle-last-batch-in-lstm-hidden-state/40858/2
        """
        return (
            torch.zeros(self.num_layers * self.directions_count, input_batch_size, self.rnn_hidden_dim).to(self.device),
            torch.zeros(self.num_layers * self.directions_count, input_batch_size, self.rnn_hidden_dim).to(self.device)
        )


    def forward(self, x):
        self.hidden_state = self.init_hidden(x.shape[0])
        x, (self.hidden_state, self.cell_state) = self.lstm(x, self.hidden_state)
        x = x.reshape(x.shape[0], -1)
        x = self.final(x)

        return x, (self.hidden_state, self.cell_state)




class Resnet18LSTM(nn.Module):
    """
    INPUT
        num_classes: The number of categories which Resnet18LSTM will classify the input into
        latent_dim: The dimension of the encoding of the InceptiveV1LSTM extractor
        lstm_layers: The lstm number stacked in the vertical direction
        hidden_dim: The dimension of hidden state
        bidirectional: uni/bi-direction of LSTM
    OUTPUT
        logist: the distribution over the num_classes categories
    """
    def __init__(self, num_classes, device, latent_dim=1000, lstm_layers=1, hidden_dim=1024, clips_num=50, bidirectional=False):
        super(Resnet18LSTM, self).__init__()
        self.prev_hidden_state = None # prev_hidden_state: the hidden state when t=1,...,seq_len-1
        self.encoder = models.resnet18(pretrained=True)
        self.lstm = LSTM(num_classes, latent_dim, lstm_layers, hidden_dim, clips_num, bidirectional, device)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x_lstm = x.view(batch_size, seq_length, -1)
        logist, (self.hidden_state, self.cell_state) = self.lstm(x_lstm)

        return logist



if __name__ == '__main__':
    batch_size, clips_num, C, W, H = 4, 50, 3, 224, 224
    input_rand = torch.rand(batch_size, clips_num, C, W, H)
    model = Resnet18LSTM(num_classes=2, device='cpu')
    output = model(input_rand)
    print(output)

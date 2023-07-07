import torch
import torch.nn as nn
class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2) # MultiheadAttention requires [seq_len, batch, features]
        out, _ = self.attention(x, x, x)
        return out.permute(1, 0, 2) # Return to original [batch, seq_len, features] shape

class StockPredictor3(nn.Module):
    """Multi step forecasting

    Args:
        pr: prediction periods.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, pr, output_dim, dropout_prob=0.2, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.pr = pr  # number of forecast days

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = MultiheadAttention(hidden_dim*2, num_heads)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim*2, output_dim * pr)
        # self.fc2 = nn.Linear(output_dim, )  # forecast pr days into the future

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device=x.device).requires_grad_()

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.attention(out)
        out = out[:, -1, :] # getting the output from the last time step
        out = self.dropout(out)
        # out = torch.tanh(self.fc1(out))
        out = self.fc1(out).view(x.size(0), self.pr)
        return out
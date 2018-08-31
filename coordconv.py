import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.ion()
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)



def generate_data():
    # generate all of the input images
    n_samples = 56 ** 2
    onehots = np.pad(np.eye(n_samples).reshape((n_samples, 1, 56, 56)), ((0,0), (0, 0), (4,4), (4,4)), "constant")
    onehots = torch.from_numpy(onehots).float()
    with torch.no_grad():
        conv = nn.Conv2d(1, 1, kernel_size=9, padding=4, stride=1)
        conv.weight.data.fill_(1)
        conv.bias.data.fill_(0)
        dataset_x = conv(onehots)
    y_i = torch.arange(56).view(56, 1).repeat(1, 56).view(-1, 1)
    y_j = torch.arange(56).repeat(56, 1).view(-1, 1)
    dataset_y = torch.cat((y_i, y_j), dim=1)

    # split the dataset tensor into train, test sets
    sample_order = np.arange(n_samples)
    
    np.random.shuffle(sample_order)
    train_idxes = sample_order[:2352]
    test_idxes = sample_order[2352:]

    train_x = dataset_x[train_idxes]
    train_y = dataset_y[train_idxes] 
    test_x = dataset_x[test_idxes]
    test_y = dataset_y[test_idxes]

    return train_x, train_y, test_x, test_y


class CoordConv2d(nn.Module):
    
    def __init__(self, device, in_channels, out_channels, kernel_size, padding, stride, input_size):
        super(CoordConv2d, self).__init__()
        self.device = device
        self.cc_xy = self.make_channels(input_size)
        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def make_channels(self, input_size):
        coord_vals = (2 * torch.arange(input_size) / input_size) - 1
        xchannel = coord_vals.repeat((input_size, 1)).unsqueeze(dim=0)
        ychannel = xchannel.permute(0, 2, 1)
        return torch.cat((xchannel.unsqueeze(dim=0), ychannel.unsqueeze(dim=0)), dim=1)

    def forward(self, x):
        n = x.shape[0]
        x = torch.cat((x, self.cc_xy.repeat(n, 1, 1, 1).to(self.device)), dim=1)
        return self.conv(x)



class CNN(nn.Module):

    def __init__(self, device, coordconv=True):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.bn2 = nn.BatchNorm2d(2)
        self.bn3 = nn.BatchNorm2d(2)
        # input (N, C, H, W) is (N, 1, 64, 64)
        if coordconv:
            self.conv1 = CoordConv2d(device, 1, 1, kernel_size=1, padding=0, stride=1, input_size=64)
        else:
            self.conv1 = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=3, padding=1, stride=2)
        self.fc1 = nn.Linear(2 * 4 * 4, 2)
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(self.pool(self.bn2(self.conv2(x))))
        x = self.relu(self.pool(self.bn3(self.conv3(x))))
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_x, train_y, test_x, test_y = generate_data()
train_x, train_y, test_x, test_y = train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)
models = [CNN(device=device, coordconv=False).to(device), CNN(device=device, coordconv=True).to(device)]
model_names = ["Standard", "CoordConv"]
n_epochs = 5000
batch_size = 32

loss_history = [[],[]]  # append validation losses as the models train for eventual plotting.

for model_idx, model in enumerate(models):
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.L1Loss()
    for epoch_i in range(n_epochs):
        model.train()
        # sample a batch
        sample_idxes = np.random.randint(2352, size=(batch_size))
        pred_y = model(train_x[sample_idxes])
        loss = criterion(pred_y, train_y[sample_idxes])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch_i + 1) % 50 == 0:
            # evaluate the model
            with torch.no_grad():
                model.eval()
                test_loss = criterion(model(test_x), test_y)
            print("Epoch: {}/{}, Model: {}, Train Loss: {:.3f}, Test Loss: {:.3f}".format(epoch_i + 1, n_epochs, model_names[model_idx], loss.item(), test_loss.item()))
            loss_history[model_idx].append(test_loss.item())

    print("---" * 10, "\n")


# plot the validation loss histories
plt.ion()
epoch_x = np.arange(50, n_epochs+1, 50)
plt.plot(epoch_x, loss_history[0], label="Standard", lw=3)
plt.plot(epoch_x, loss_history[1], label="CoordConv", lw=3)
plt.legend()
plt.title("Supervised Regression Error")
plt.xlabel("Epoch")
plt.ylabel("Error")


# Functions to show the CoordConv predictions on input images after training, if desired.
def test_show(sample_idx):
    plt.imshow(test_x[sample_idx, 0].detach().cpu().numpy(), cmap='Reds')
    with torch.no_grad():
        pred_y = model(test_x)[sample_idx]
    print("Predicted: {}, Target: {}".format(pred_y.detach().cpu().numpy(), test_y[sample_idx].detach().cpu().numpy()))

def train_show(sample_idx):
    plt.imshow(train_x[sample_idx, 0].detach().cpu().numpy(), cmap='Reds')
    with torch.no_grad():
        model.eval()
        pred_y = model(train_x)[sample_idx]
    print("Predicted: {}, Target: {}".format(pred_y.detach().cpu().numpy(), train_y[sample_idx].detach().cpu().numpy()))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
from CMap2D import gridshow, CMap2D

from GNN.STGNNs import A3T_GNN
from GNN.data_loader import HumanPresenceDataLoader

def run_A3T_GNN(test_loader, train_loader, static_edge_index):
    # GPU support
    # DEVICE = torch.device('cuda') # cuda
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model and optimizers
    model = A3T_GNN(node_features=2, periods=12, batch_size=batch_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print_model_layout(model, optimizer)

    for epoch in range(5):
        step = 0
        loss_list = []
        for encoder_inputs, labels in train_loader:
            y_hat = model(encoder_inputs, static_edge_index)         # Get model predictions
            loss = loss_fn(y_hat, labels) # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step= step+ 1
            loss_list.append(loss.item())
            if step % 100 == 0 :
                print(sum(loss_list)/len(loss_list))
        print("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))

    yhat = evaluate_performance(model, loss_fn, static_edge_index, test_loader)

    plot_occupancy(yhat, labels)


def reshape_data(dataset, batch_size):
    # GPU support
    # DEVICE = torch.device('cuda') # cuda
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shuffle = True

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    print("Number of train buckets: ", len(list(train_dataset)))
    print("Number of test buckets: ", len(list(test_dataset)))

    train_input = np.array(train_dataset.features)  # (27399, 207, 2, 12)
    train_target = np.array(train_dataset.targets)  # (27399, 207, 12)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=True)

    test_input = np.array(test_dataset.features)  # (, 207, 2, 12)
    test_target = np.array(test_dataset.targets)  # (, 207, 12)
    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)


    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(DEVICE)
        break

    return test_loader, train_loader, static_edge_index


def print_model_layout(model, optimizer):
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)
    #--------------------------------------------------
    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])


def evaluate_performance(model, loss_fn, static_edge_index, test_loader):
    model.eval()
    step = 0
    # Store for analysis
    total_loss = []
    for encoder_inputs, labels in test_loader:
        # Get model predictions
        yhat = model(encoder_inputs, static_edge_index)
        # Mean squared error
        loss = loss_fn(yhat, labels)
        total_loss.append(loss.item())
        # Store for analysis below
        # test_labels.append(labels)
        # predictions.append(y_hat)

    print("Test MSE: {:.4f}".format(sum(total_loss) / len(total_loss)))

    return yhat


def plot_occupancy(y_hat, labels):
    yhat1 = reshape_output(y_hat, t=11, batch_nr=10)
    ytrue1 = reshape_output(labels, t=11, batch_nr=10)

    grid1 = CMap2D()
    grid2 = CMap2D()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    grid1.from_array(yhat1, origin=(0, 0), resolution=1.)
    plt.sca(ax1)
    plt.title("Predicted Human Presence")
    gridshow(grid1.occupancy())

    grid2.from_array(ytrue1, origin=(0, 0), resolution=1.)
    plt.sca(ax2)
    plt.title("True Human Presence")
    gridshow(grid2.occupancy())

    plt.show()


def reshape_output(arr, t, batch_nr):
    y = arr.cpu().detach().numpy()
    y = y[batch_nr][:, t]
    y = np.array(np.array_split(y, 100))

    return y


if __name__ == "__main__":
    loader = HumanPresenceDataLoader()
    dataset = loader.get_dataset(num_t_in=12, num_t_out=12)
    print(next(iter(dataset)))

    batch_size = 16
    test_loader, train_loader, static_edge_index = reshape_data(dataset, batch_size)

    run_A3T_GNN(test_loader, train_loader, static_edge_index)
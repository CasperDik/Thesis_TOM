import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
from CMap2D import gridshow, CMap2D

from GNN.STGNNs import A3T_GNN
from GNN.data_loader import HumanPresenceDataLoader
import pickle

def run_A3T_GNN(test_loader, train_loader, static_edge_index, idx, threshold):
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
            step = step+1
            loss_list.append(loss.item())
            if step % 100 == 0 :
                print(sum(loss_list)/len(loss_list))
        print("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))

    yhat, labels = evaluate_performance(model, loss_fn, static_edge_index, test_loader, threshold)

    plot_occupancy_grid(yhat, labels, t=1, batch_nr=5, threshold=threshold, idx=idx)


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


def plot_occupancy_grid(y_hat, labels, t, batch_nr, threshold, idx):
    y_hat = y_hat.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    y_hat = back_to_occupancy_grid(y_hat, idx, batch_nr=batch_nr, t=t)
    labels = back_to_occupancy_grid(labels, idx, batch_nr=batch_nr, t=t)

    yhat1 = reshape_output(y_hat, t=t, threshold=threshold)
    ytrue1 = reshape_output(labels, t=t, threshold=threshold)

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

    # plt.savefig("output.png")


def reshape_output(y, t, threshold):
  y = y[:, t]
  y = np.where(y>threshold, y, 0)
  y = np.array(np.array_split(y, 100))

  return y


def back_to_occupancy_grid(F, idx, batch_nr, t):
    """add back the removed nodes"""
    full_F = np.insert(F[batch_nr][:,], idx[0] - np.arange(len(idx[0])), 0, axis=0)
    return full_F


def evaluate_performance(model, loss_fn, static_edge_index, test_loader, threshold):
    model.eval()
    step = 0
    # Store for analysis
    total_loss = []
    total_precision = []
    total_recall = []
    total_accuracy = []
    total_f1 = []

    # todo: add the performance metrics and test it

    for encoder_inputs, labels in test_loader:
        # Get model predictions
        yhat = model(encoder_inputs, static_edge_index)

        # Mean squared error
        loss = loss_fn(yhat, labels)
        total_loss.append(loss.item())

        # use threshold to make predictions binary
        yhat = np.where(yhat > threshold, 1, 0)
        labels = np.where(labels > threshold, 1, 0)

        # precision --> true positives / all positives of prediction
        tp = len(np.where((yhat == labels) & (yhat == 1))[0])
        ap = len(np.where(yhat == 1)[0])
        precision = tp / ap
        total_precision.append(precision)

        # recall --> tp/(tp+fn)
        fn = len(np.where((yhat != labels) & (yhat == 0))[0])
        recall = tp / (tp + fn)
        total_recall.append(recall)

        # accuracy --> (tp+tn) / (tp+tn+fp+fn)
        tn = len(np.where((yhat == labels) & (yhat == 0))[0])
        fp = len(np.where((yhat != labels) & (yhat == 1))[0])
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        total_accuracy.append(accuracy)

        # F1 score --> 2 * (precision*recall)/(precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        total_f1.append(f1)

    print("Test MSE: {:.4f}".format(sum(total_loss) / len(total_loss)))
    print("Test accuracy: {:.4f}".format(sum(total_accuracy) / len(total_accuracy)))
    print("Test precision: {:.4f}".format(sum(total_precision) / len(total_precision)))
    print("Test recall: {:.4f}".format(sum(total_recall) / len(total_recall)))
    print("Test F1: {:.4f}".format(sum(total_f1) / len(total_f1)))

    return yhat, labels     # todo: this returns the last yhat, should that be the case??


def inside_model_performance_loop(yhat, labels, threshold):
    loss_fn = torch.nn.MSELoss()

    loss = loss_fn(yhat, labels)
    print(loss)

    # make it binary with threshold
    yhat = np.where(yhat > threshold, 1, 0)
    labels = np.where(labels > threshold, 1, 0)

    # precision --> true positives / all positives of prediction
    tp = len(np.where((yhat == labels) & (yhat == 1))[0])
    ap = len(np.where(yhat == 1)[0])
    precision = tp/ap
    print("precision: ", precision)

    # recall --> tp/(tp+fn)
    fn = len(np.where((yhat != labels) & (yhat == 0))[0])
    recall = tp/(tp+fn)
    print("recall: ", recall)

    # accuracy --> (tp+tn) / (tp+tn+fp+fn)
    tn = len(np.where((yhat == labels) & (yhat == 0))[0])
    fp = len(np.where((yhat != labels) & (yhat == 1))[0])
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print("accuracy: ", accuracy)

    # F1 score --> 2 * (precision*recall)/(precision + recall)
    F1 = 2*(precision*recall)/(precision+recall)
    print("F1: ", F1)


if __name__ == "__main__":
    # idx = np.load("data/input_matrices/idx.npy")
    # F = np.load("data/input_matrices/FeatureMatrix.npy")
    # A = np.load("data/input_matrices/Adj_Matrix.npy")
    #
    # loader = HumanPresenceDataLoader(A, F)
    # dataset = loader.get_dataset(num_t_in=12, num_t_out=12)
    # print(next(iter(dataset)))
    #
    # batch_size = 16
    # test_loader, train_loader, static_edge_index = reshape_data(dataset, batch_size)
    #
    # run_A3T_GNN(test_loader, train_loader, static_edge_index, idx)
    y_hat = pickle.load(open("data/datasets/yhat.p", "rb"))
    labels = pickle.load(open("data/datasets/labels.p", "rb"))
    # todo: klopt labels wel? is het normalized?

    inside_model_performance_loop(y_hat, labels, threshold=0.7)


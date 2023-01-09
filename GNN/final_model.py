import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2, TGCN2


class STGNN_model():
    def __init__(self):
        self.A, _ = dense_to_sparse(torch.from_numpy(np.load(r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\input_matrices\Adj_Matrix_Reduced.npy")))  # change location later
        self.idx = np.load(r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\input_matrices\idx.npy")
        self.stgnn = TemporalGNN(node_features=4, periods_in=5, periods_out=20, batch_size=32, num_edges=self.A.shape[1])
        self.stgnn = torch.load(r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\model.pt", map_location=torch.device('cpu'))
        self.stgnn.eval()

    def run_model(self, paths_inputs:list):
        self.F = self.create_feature_matrix(paths_inputs)

        self.process_data()
        self.yhat = self.get_outputs()

        return self.yhat

    def create_feature_matrix(self, paths_inputs:list):
        paths_inputs = self.check_time_in(paths_inputs)

        f0 = self.load_txt_to_array(paths_inputs[0])
        f1 = self.load_txt_to_array(paths_inputs[1])
        F = np.stack((f0, f1))

        for txt in paths_inputs[2:]:
            f = self.load_txt_to_array(txt)
            F = np.vstack((F, f[None, :, :]))

        return F

    def load_txt_to_array(self, txt:str):
        if txt[-4:] == ".txt":
            self.delete_hashtag(txt)
            d = np.genfromtxt(txt, delimiter=[1, 20], dtype=[("f0", np.uint8), ("f1", object)]) # open as np array
            self.check_resolution(d)
            d = self.load_features(d)
        else:
            raise SystemExit("Filetype unsupported. All input files must be .txt")
        return d.astype('uint8')

    def delete_hashtag(self, f):
        """remove the # from the input files, otherwise will stop reading after # at later stage"""
        with open(f, "rb") as input_file:
            s = input_file.read()
            input_file.close()
            s = s.replace(b"#", b"")

        with open(f, "wb") as output_file:
            output_file.write(s)

    def load_features(self, d):
        x = d["f1"].astype("U")

        w = np.where(np.char.find(x, "Wall") > 0, 1, 0)
        c = np.where(np.char.find(x, "coffee") > 0, 1, 0)
        ws = np.where(np.char.find(x, "WS") > 0, 1, 0)

        # first column is human presence, second wall, third coffee, fourth workstation
        d = np.stack((d["f0"], w), axis=1)
        d = np.concatenate((d, c[:, None]), axis=1)
        d = np.concatenate((d, ws[:, None]), axis=1)
        return d

    def check_resolution(self, d):
        # check if resolutions is correct --> must have 10000 cells
        nr_nodes = 10000
        if d.shape[0] != nr_nodes:
            raise SystemExit("Resolution unsupported by model. Resolution must be 100x100")

    def check_time_in(self, paths_inputs:list):
        time_in = 5
        if len(paths_inputs) != time_in:
            print("The model uses a historical time series of length ", time_in, " but ", len(paths_inputs), " files were supplied.")
            print("First ", time_in-len(paths_inputs), " will be dropped.")
            paths_inputs = paths_inputs[-5:]
        return paths_inputs

    def process_data(self):
        # drop nodes
        self.F = np.delete(self.F, self.idx, axis=1)
        # normalize
        self.F = self.normalize_zscore(self.F)


    def normalize_zscore(self, F):
        F = F.transpose((1, 2, 0))
        means = np.mean(F, axis=(0, 2))
        F = F - means.reshape(1, -1, 1)
        stds = np.std(F, axis=(0, 2))
        F = F / stds.reshape(1, -1, 1)
        F = np.expand_dims(F, axis=0)

        return F

    def get_outputs(self):
        with torch.no_grad():
            yhat = self.stgnn(torch.from_numpy(self.F.astype(np.float32)), self.A)
        yhat = np.insert(yhat, self.idx[0] - np.arange(len(self.idx[0])), 0, axis=1)
        return yhat

    def export_as_txt(self, yhat: np.ndarray, path: str = "", regression_output: bool = True, classification_output: bool = False):
        for i in range(yhat.shape[2]):
            if regression_output==True:
                np.savetxt(path + "heatmap_reg_t" + str(i+1) + ".txt", yhat[0, :, i])
            if classification_output==True:
                threshold = 0.16
                yhat = np.where(yhat > threshold, 1, 0).astype("uint8")
                np.savetxt(path + "heatmap_class_t" + str(i + 1) + ".txt", yhat[0, :, i], )

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods_in, periods_out, batch_size, num_edges):
        super(TemporalGNN, self).__init__()

        # initialize learnable edge weights if num_edge is provided
        # self.edge_weight = torch.nn.Parameter(torch.ones(num_edges))
        self.edge_weight = torch.nn.Parameter(torch.full((num_edges,), 1 / 8))

        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=8, periods=periods_in, batch_size=batch_size)

        # additonal hidden layer
        self.hlayer = torch.nn.Linear(16, periods_out * 2)

        # Equals single-shot prediction
        self.linear = torch.nn.Linear(8, periods_out)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, self.edge_weight.relu())
        # print(self.edge_weight.std())
        h = F.relu(h)
        # h = F.dropout(h, p=0.4, training=self.training)
        # h = self.hlayer(h)
        # h = F.dropout(h, p=0.4, training=self.training)
        h = self.linear(h)

        return h

if __name__ == "__main__":
    # todo: add cuda support

    model = STGNN_model()
    txts = [r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_22-04-22\simulation1-50p-100cm\heatmap_08H57m32s.txt", r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_22-04-22\simulation1-50p-100cm\heatmap_08H57m33s.txt", r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_22-04-22\simulation1-50p-100cm\heatmap_08H57m34s.txt", r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_22-04-22\simulation1-50p-100cm\heatmap_08H57m35s.txt", r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_22-04-22\simulation1-50p-100cm\heatmap_08H57m36s.txt"]
    yhat = model.run_model(paths_inputs=txts)
    model.export_as_txt(yhat, path="data/output/", regression_output=False, classification_output=True)



    # todo: change model file and class
    # todo: change threshold, time_in, time_out

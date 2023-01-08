import numpy as np
import pickle

yhat = pickle.load(open(r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\datasets\yhat.p", "rb"))
labels = pickle.load(open(r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\datasets\labels.p", "rb"))

threshold = 0.5
periods_out = 12
r_precision = {}

for encoder_inputs, labels in test_loader:
    y_hat = np.where(yhat > threshold, 1, 0)
    labels_ = np.where(labels > threshold, 1, 0)

    for t in range(periods_out):
        r_precision[t] = []
        tp = len(np.where((y_hat[:, :, t] == labels_[:, :, t]) & (y_hat[:, :, t] == 1))[0])
        ap = len(np.where(y_hat[:, :, t] == 1)[0])
        precision = tp / ap
        r_precision[t].append(precision)

for i in range(periods_out):
    r_precision[i] = sum(r_precision[i]) / len(r_precision[i])

print("Average precision is: {.4f}".format(sum(list(r_precision.values()))/len(list(r_precision.values()))))

print("t", "precision")
for k,v in r_precision.items():
    print(k, round(v, 3))

print("yes")
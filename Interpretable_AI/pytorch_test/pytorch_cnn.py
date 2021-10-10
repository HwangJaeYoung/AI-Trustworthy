import torch
from functools import partial
from sklearn.datasets import make_classification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

X, y = make_classification(n_samples=100, n_features=5, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = X[idx]
        label = y[idx]

        if self.transform:
            x = self.transform(x)
            label = self.transform(label)

        return x, label

torch_tensor_float32 = partial(torch.tensor, dtype=torch.float32)
transformed_dataset = CustomDataset(X, y, transform=torch_tensor_float32)

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True)

num_features = 5
num_outputs = 1
layer_dims = [num_features, 5, 3, num_outputs]

# Model Definition
model = torch.nn.Sequential(
    torch.nn.Linear(5, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1),
    torch.nn.Sigmoid()
)

class BinaryClassifier(torch.nn.Sequential):
    def __init__(self, layer_dims):
        super(BinaryClassifier, self).__init__()

        for idx, dim in enumerate(layer_dims):
            if idx < len(layer_dims) - 1:
                module = torch.nn.Linear(dim, layer_dims[idx + 1])
                self.add_module(f"linear{idx}", module)
            if idx < len(layer_dims) - 2:
                activation = torch.nn.ReLU()
                self.add_module(f"relu{idx}", activation)
            elif idx == len(layer_dims) - 2:
                activation = torch.nn.Sigmoid()
                self.add_module(f"sigmoid{idx}", activation)

bc_model = BinaryClassifier(layer_dims)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(bc_model.parameters())

num_epochs = 10

for epoch in range(num_epochs):
    for idx, (X_batch, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = bc_model(X_batch)
        outputs = torch.flatten(outputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

pred_var = bc_model(transformed_dataset[0][0])
print(pred_var.detach().numpy()[0])
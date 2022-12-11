import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda

from quickstart import NeuralNetwork, train, test
from datasets_dataloaders import CustomImageDataset

device = "cpu"
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def load_from_mydata():
    return DataLoader(CustomImageDataset(
        "./data/mydata/annotation.csv",
        "data/mydata/images",
        # ToTensor(),
        # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    ),
        batch_size=64
    )


train_dataloader = load_from_mydata()
test_dataloader = load_from_mydata()

epochs = 5
for t in range(epochs):
    print(f"epoch: {t}")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)

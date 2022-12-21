import torch
seed = 42
torch.manual_seed(seed)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda

from quickstart import NeuralNetwork, train, test, show_dataloader
from datasets_dataloaders import CustomImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
#model = NeuralNetwork(1280*720*3).to(device)
img_size = 512
img_channel = 3
model = NeuralNetwork(img_size*img_size*img_channel, out_features=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

data_dir = f"resized_{img_size}_{img_size}"

mydata = CustomImageDataset(
        f"./data/mydata/annotation_0_1.csv",
        f"data/mydata/{data_dir}",
        # ToTensor(),
        transform=Lambda(lambda x: x.to(torch.float32))
        # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

def load_from_mydata():
    return DataLoader(mydata,
        batch_size=128,
        shuffle=True
    )


train_dataloader = load_from_mydata()
test_dataloader = load_from_mydata()

show_dataloader(test_dataloader)

epochs = 20
for t in range(epochs):
    print(f"epoch: {t+1}/{epochs}")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)

model.eval()


for i in range(280,300):
    x, y = mydata[i][0], mydata[i][1]
    with torch.no_grad():
        # print("------")
        # print(f"i: {i}")
        # print(f"x.shape: {x.shape}")
        x_as_one = x.unsqueeze(dim=0).to(device)
        # print(f"x_as_one.shape: {x_as_one.shape}")
        # print(y)
        pred = model(x_as_one)
        predicted = pred[0].argmax(0)
        print(f'{pred[0]} Predicted: "{predicted}", Actual: "{y}" in i: {i}')
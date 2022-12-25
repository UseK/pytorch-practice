from datasets_dataloaders import CustomImageDataset
from quickstart import CNN, NeuralNetwork, train, test, show_dataloader
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
from torch import nn
import torch
seed = 42
torch.manual_seed(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
#model = NeuralNetwork(1280*720*3).to(device)
img_size = 512
img_channel = 3

# annotations_path = "./data/mydata/annotation_0_1.csv"
# data_dir = f"resized_{img_size}_{img_size}"
# model = NeuralNetwork(img_size*img_size*img_channel, out_features=2).to(device)

# annotations_path = "./data/mydata/annotation_0_1.csv"
# data_dir = "crop_600_30_670_70" # (670 - 600) * (70 - 30) = 2800
# model = NeuralNetwork(img_channel * 70 * 40, out_features=2).to(device)

# model = NeuralNetwork(img_size * img_size * img_channel, out_features=2).to(device)
model = CNN()
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


moviedata = CustomImageDataset(
    annotations_file="./data/mydata/annotation_movie.csv",
    img_dir="data/movies/2022121721104602_resized_512_512",
    transform=Lambda(lambda x: x.to(torch.float32))
)

clipdata = CustomImageDataset(
    annotations_file="./data/mydata/annotation_0_1.csv",
    img_dir="data/mydata/resized_512_512",
    transform=Lambda(lambda x: x.to(torch.float32))
)


def load_from_mydata(dataset: CustomImageDataset):
    return DataLoader(dataset,
                      batch_size=32,
                      shuffle=True
                      )


train_dataloader = load_from_mydata(moviedata)
test_dataloader = load_from_mydata(clipdata)

show_dataloader(test_dataloader)

epochs = 1
for t in range(epochs):
    print(f"epoch: {t+1}/{epochs}")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)

model.eval()


for i in range(280, 300):
    x, y = clipdata[i][0], clipdata[i][1]
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

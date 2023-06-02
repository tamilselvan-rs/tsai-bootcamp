import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary

def get_correct_predict_count(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def plot_data(data_loader):
    batch_data, batch_label = next(iter(data_loader)) 

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def is_cuda_available():
   return torch.cuda.is_available()

def get_dst_device():
    return torch.device("cuda" if is_cuda_available() else "cpu")

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def setup_train_loader(destination):
    batch_size = 512
    kwargs = {
        'batch_size': batch_size, 
        'shuffle': True, 
        'num_workers': 2, 
        'pin_memory': True
    }
    train_data = datasets.MNIST(destination, train=True, download=True, transform=get_train_transforms())
    return torch.utils.data.DataLoader(train_data, **kwargs)

def setup_test_loader(destination):
    batch_size = 512
    kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True
    }
    test_data = datasets.MNIST(destination, train=False, download=True, transform=get_test_transforms())
    return torch.utils.data.DataLoader(test_data, **kwargs)

def train_model(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += get_correct_predict_count(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_accuracy = 100 * correct / processed
    train_loss /= len(train_loader)

  return [ train_accuracy, train_loss ]


def test_model(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += get_correct_predict_count(output, target)


    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return [test_accuracy, test_loss]

def plot_results(train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def print_model_summary(model):
    summary(model, input_size=(1, 28, 28)) 
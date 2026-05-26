import torch
import torch.nn as nn
from train import SimpleNet, correct
import os

model = SimpleNet()
model.load_state_dict(torch.load('model_fp32.pth'))
model.eval()

print("Original model loaded successfully")

size_fp32 = os.path.getsize('model_fp32.pth')/1024
print(f"FP32 model size: {size_fp32:.1f} KB")


model_quantised = torch.quantization.quantize_dynamic(
       model,
     {nn.Linear},
     dtype = torch.qint8
 )

torch.save(model_quantised.state_dict(), 'model_int8.pth')
size_int8 = os.path.getsize('model_int8.pth')/1024
print(f"INT8 model size: {size_int8:.1f} KB")
print(f"Size reduction: {size_fp32/size_int8:.1f} x smaller")

from torchvision import datasets, transforms
from train import SimpleNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.MNIST(root ='./data', train = False, download = False, transform = transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_quantised(images)
        predicted = torch.argmax(outputs, dim = 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

quant_accuracy = 100 * correct / total
print(f"Quantised model accuracy: {quant_accuracy:.2f}%")
print(f"Accuracy drop: {92.33 - quant_accuracy:.2f}%")

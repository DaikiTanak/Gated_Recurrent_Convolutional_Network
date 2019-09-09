# Gated Recurrent Convolutional Network (pytorch version)

## Description
Unofficial pytorch implementation of Gated Recurrent Convolution Neural Network.

The original Paper is **"Gated Recurrent Convolution Neural Network for
OCR" (NIPS 2017)**. Link: http://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf

## Requirement
```  
torch                              1.0.0     
Python                             3.7.0
```

## Usage
This code assumes that input images' height is 32 pixels or 64 pixels. 


```python
import torch
from model import GatedRecurrentConvNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# define model
model = GatedRecurrentConvNet(in_channels=3) # if input image is RGB
# model = GatedRecurrentConvNet(in_channels=1) # if input image is Grayscale
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# get_dataloader() is to be implemented
train_loader, val_loader = get_dataloader()

# Maxiumum length of input sequence
seq_max_length = 10

# CTC loss can be used for text recognition
loss_func = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

log_softmax = torch.nn.functional.log_softmax

while True:

    model.train()
    for idx, (img, text_label) in enumerate(train_loader):
        img = img.to(device)

        # forwarding
        output = model(img)
        output = log_softmax(output, dim=2)

        # convert_text_label() is to be implemented
        sparse_label = convert_text_label(text_label, max_length=seq_max_length, blank=0)
        sparse_label = sparse_label.to(device)

        input_lens = torch.IntTensor(img.size(0)*[output.size(1)])
        target_lens = torch.IntTensor([len(t) for t in text_label])

        # CTC loss
        # log_probs : (input_length, batch_size, num classes)
        # target : (batch_size, max_target_length)
        loss = loss_func(output.permute(1, 0, 2), label, input_lens, target_lens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

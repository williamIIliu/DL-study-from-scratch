import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,dataset


from torch.autograd import Variable

import numpy as np
import math
# from lib.utils import Dataset, masking_noise
# from lib.ops import MSELoss, BCELoss

def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, out_features, activation="relu",
                 dropout=0.2, tied=False):
        super(self.__class__, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if tied:
            self.deweight = self.weight.t()
        else:
            self.deweight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.vbias = Parameter(torch.Tensor(in_features))

        if activation == "relu":
            self.enc_act_func = nn.ReLU()
        elif activation == "sigmoid":
            self.enc_act_func = nn.Sigmoid()
        elif activation == "none":
            self.enc_act_func = None
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # initialize parameters
        stdv = 0.01
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        stdv = 0.01
        self.deweight.data.uniform_(-stdv, stdv)
        self.vbias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.enc_act_func is not None:
            return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))
        else:
            return self.dropout(F.linear(x, self.weight, self.bias))

    def encode(self, x, train=True):
        if train:
            self.dropout.train()
        else:
            self.dropout.eval()
        if self.enc_act_func is not None:
            return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))
        else:
            return self.dropout(F.linear(x, self.weight, self.bias))

    def encodeBatch(self, dataloader):
        use_cuda = torch.cuda.is_available()
        encoded = []
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            # print(inputs.size())
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            hidden = self.encode(inputs, train=False)
            encoded.append(hidden.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def decode(self, x, binary=False):
        if not binary:
            return F.linear(x, self.deweight, self.vbias)
        else:
            return F.sigmoid(F.linear(x, self.deweight, self.vbias))

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.3,
            loss_type="mse"):
        """
        data_x: FloatTensor
        valid_x: FloatTensor
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Denoising Autoencoding layer=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.6)
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "cross-entropy":
            criterion = nn.BCELoss()

        # validate
        total_loss = 0.0
        total_num = 0
        for batch_idx, (inputs, _) in enumerate(validloader):
            # next 3 lines can be canceled
            # inputs = inputs.view(inputs.size(0), -1).float()
            # if use_cuda:
            #     inputs = inputs.cuda()
            inputs = Variable(inputs)
            # print(inputs.size())
            hidden = self.encode(inputs)
            if loss_type == "cross-entropy":
                outputs = self.decode(hidden, binary=True)
            else:
                outputs = self.decode(hidden)
            # print(outputs.size())
            valid_recon_loss = criterion(outputs, inputs)
            total_loss += valid_recon_loss.data * len(inputs)
            total_num += inputs.size()[0]

        valid_loss = total_loss / total_num
        print("#Epoch 0: Valid Reconstruct Loss: %.4f" % (valid_loss))

        self.train()
        for epoch in range(num_epochs):
            # train 1 epoch
            train_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                # inputs = inputs.view(inputs.size(0), -1).float()
                inputs_corr = masking_noise(inputs, corrupt)
                # if use_cuda:
                #     inputs = inputs.cuda()
                #     inputs_corr = inputs_corr.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                inputs_corr = Variable(inputs_corr)

                hidden = self.encode(inputs_corr)
                if loss_type == "cross-entropy":
                    outputs = self.decode(hidden, binary=True)
                else:
                    outputs = self.decode(hidden)

                recon_loss = criterion(outputs,inputs)
                train_loss += recon_loss.data * len(inputs)
                recon_loss.backward()
                optimizer.step()

            # validate
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                # inputs = inputs.view(inputs.size(0), -1).float()
                # if use_cuda:
                #     inputs = inputs.cuda()
                inputs = Variable(inputs)
                hidden = self.encode(inputs, train=False)
                if loss_type == "cross-entropy":
                    outputs = self.decode(hidden, binary=True)
                else:
                    outputs = self.decode(hidden)

                valid_recon_loss = criterion(outputs, inputs)
                valid_loss += valid_recon_loss.data * len(inputs)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print("#Epoch %3d: Reconstruct Loss: %.4f, Valid Reconstruct Loss: %.4f, Learning rate: %.2e" % (
                epoch + 1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), current_lr))


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

if __name__ == '__main__':
    device = 'cuda'
    # check data
    train_x = np.load('../../dataset/JTEXT/train_X.npy')
    train_y = np.load('../../dataset/JTEXT/train_Y.npy')
    test_x = np.load('../../dataset/JTEXT/test_X.npy')
    test_y = np.load('../../dataset/JTEXT/test_Y.npy')
    X_train = torch.tensor(train_x, dtype=torch.float32).to(device)
    Y_train = torch.tensor(train_y, dtype=torch.float32).to(device)
    X_test = torch.tensor(train_x, dtype=torch.float32).to(device)
    Y_test = torch.tensor(train_y, dtype=torch.float32).to(device)
    X_train = X_train.view(X_train.size(0), -1)
    X_test = X_test.view(X_test.size(0),-1)
    train_dataset = dataset.TensorDataset(X_train, Y_train)
    val_dataset = dataset.TensorDataset(X_train, Y_train)
    train_iter = DataLoader(train_dataset,
                            batch_size=256,
                            shuffle=True,
                            num_workers=0)
    test_dataset = dataset.TensorDataset(X_test, Y_test)
    test_iter = DataLoader(val_dataset,
                           batch_size=256,
                           shuffle=True,
                           num_workers=0)

    # optimization
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
    # # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.6)

    # test_x = torch.randn((128,100)).to(device)
    # test_batch = torch.randn((2,128,1024))
    # print(test_x.size())

    # build network
    DAE = DenoisingAutoencoder(14400,1024,'sigmoid').to(device)
    print(DAE)

    # encode
    # encoded_batch = DAE.encodeBatch(train_iter)
    # print(encoded_batch.size())

    # fit
    DAE.fit(train_iter,test_iter,lr=1e-1,batch_size=258,num_epochs=100,corrupt=0.3,loss_type='mse') #cross-entropy would use different reconstruction loss sunc
    # sDAE = StackedAutoencoder(1024,[512,256,128],dropout=0.25)
    # print(sDAE)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from DAE import DenoisingAutoencoder as DenoisingAutoencoder
from torch.autograd import Variable
from torch.utils.data import DataLoader,dataset
def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise

class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.labels = self.labels.cuda()

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def buildNetwork(layers_dim, activation="relu", dropout=0.):
    net = []
    for i in range(1, len(layers_dim)):
        net.append(nn.Linear(layers_dim[i-1], layers_dim[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0.:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)



class StackedDAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
                 encodeLayer=[400], decodeLayer=[400], activation="relu",
                 dropout=0.0, tied=False):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x):
        loss = -torch.mean(torch.sum(x * torch.log(torch.clamp(recon_x, min=1e-10)) +
                                     (1 - x) * torch.log(torch.clamp(1 - recon_x, min=1e-10)), 1))

        return loss

    def extrac_features(self,train_loader,train=False):
        # if train:
        #     self.dropout.train()
        # else:
        #     self.dropout.eval()
        #     print(self.dropout)
        use_cuda = torch.cuda.is_available()
        encoded = []
        label = []
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.view(inputs.size(0), -1).float()
            # print(inputs.size())
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            hidden = self.encoder(inputs)
            encoded.append(hidden.data)
            label.append(labels)

        encoded = torch.cat(encoded, dim=0)
        label = torch.cat(label,dim=0)
        reconstructed_dataset = dataset.TensorDataset(encoded, label)
        return reconstructed_dataset

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)

        return z, self.decode(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def pretrain(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.2,
                 loss_type="cross-entropy"):
        trloader = trainloader
        valoader = validloader
        daeLayers = []
        print("=====Stacked Denoising Autoencoding Layer Pre-Train=======")
        for l in range(1, len(self.layers)):
            f'{l} layer is training now'
            infeatures = self.layers[l - 1]
            outfeatures = self.layers[l]
            if l != len(self.layers) - 1:
                dae = DenoisingAutoencoder(infeatures, outfeatures, activation=self.activation, dropout=corrupt)
            else:
                dae = DenoisingAutoencoder(infeatures, outfeatures, activation="none", dropout=0)
            print(dae)
            if l == 1:
                dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt,
                        loss_type=loss_type)
            else:
                if self.activation == "sigmoid":
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt,
                            loss_type="cross-entropy")
                else:
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt,
                            loss_type="mse")
            data_x = dae.encodeBatch(trloader)
            valid_x = dae.encodeBatch(valoader)
            trainset = Dataset(data_x, data_x)
            trloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=0)
            validset = Dataset(valid_x, valid_x)
            valoader = torch.utils.data.DataLoader(
                validset, batch_size=1000, shuffle=False, num_workers=0)
            daeLayers.append(dae)

        self.copyParam(daeLayers)
        # it has copied the params in the net

    def copyParam(self, daeLayers):
        if self.dropout == 0:
            every = 2
        else:
            every = 3
        # input layer
        # copy encoder weight
        self.encoder[0].weight.data.copy_(daeLayers[0].weight.data)
        self.encoder[0].bias.data.copy_(daeLayers[0].bias.data)
        self._dec.weight.data.copy_(daeLayers[0].deweight.data)
        self._dec.bias.data.copy_(daeLayers[0].vbias.data)

        # print(self.encoder)
        # print(self.decoder)

        for l in range(1, len(self.layers) - every):
            # copy encoder weight
            self.encoder[l * every].weight.data.copy_(daeLayers[l].weight.data)
            self.encoder[l * every].bias.data.copy_(daeLayers[l].bias.data)

            # copy decoder weight
            self.decoder[-(l - 1) * every - 2].weight.data.copy_(daeLayers[l].deweight.data)
            self.decoder[-(l - 1) * every - 2].bias.data.copy_(daeLayers[l].vbias.data)

        # z layer
        self._enc_mu.weight.data.copy_(daeLayers[-1].weight.data)
        self._enc_mu.bias.data.copy_(daeLayers[-1].bias.data)
        self.decoder[0].weight.data.copy_(daeLayers[-1].deweight.data)
        self.decoder[0].bias.data.copy_(daeLayers[-1].vbias.data)

    def fit(self, trainloader, validloader, lr=0.001, num_epochs=10, corrupt=0.3,
            loss_type="mse"):
        """
        data_x: FloatTensor
        valid_x: FloatTensor
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Stacked Denoising Autoencoding Layer Fine-Tuning=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.6)
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "cross-entropy":
            criterion = nn.BCELoss()

        # validate
        total_loss = 0.0
        total_num = 0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs = self.forward(inputs)

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
                inputs = inputs.view(inputs.size(0), -1).float()
                inputs_corr = masking_noise(inputs, corrupt)
                if use_cuda:
                    inputs = inputs.cuda()
                    inputs_corr = inputs_corr.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                inputs_corr = Variable(inputs_corr)

                z, outputs = self.forward(inputs_corr)
                recon_loss = criterion(outputs, inputs)
                train_loss += recon_loss.data * len(inputs)
                recon_loss.backward()
                optimizer.step()

            # validate
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs = self.forward(inputs)

                valid_recon_loss = criterion(outputs, inputs)
                valid_loss += valid_recon_loss.data * len(inputs)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print("#Epoch %3d: Reconstruct Loss: %.4f, Valid Reconstruct Loss: %.4f, Learning rate: %.2e" % (
                epoch + 1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), current_lr))



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
    X_test = X_test.view(X_test.size(0), -1)
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

    # build network
    # linear = buildNetwork([128, 64, 32], 'sigmoid', 0.2)
    # print(linear)
    SDAE = StackedDAE(14400,512,False,[4096,1024], [1024,4096], 'sigmoid',0.25).to(device)
    print(SDAE)
    print(SDAE.encoder[0].weight)

    # pre-Train for better init weights
    # SDAE.pretrain(train_iter,test_iter,1e-3,256,10,0.2,'mse',)

    # check the init weights
    # print(SDAE.encoder[0].weight)
    # SDAE.save_model('../../logs/SDAE/SDAE_pre-trained.pth')

    # fit or fine-tuning the pre-trained weights because it is trained alone in that process
    SDAE.load_model('../../logs/SDAE/SDAE_pre-trained.pth')
    SDAE.fit(train_iter,test_iter,1e-3,10,0,'mse') # in fit phase, no corruption to the net

    # save wights
    SDAE.save_model('../../logs/SDAE/SDAE.pth')
    SDAE.load_model('../../logs/SDAE/SDAE.pth')
    #
    # # extract features
    # reconstructed_set = SDAE.extrac_features(train_iter)
    # print(reconstructed_set[0][0].size())
    # reconstructed_train_iter = DataLoader(reconstructed_set,
    #                         batch_size=256,
    #                         shuffle=True,
    #                         num_workers=0)



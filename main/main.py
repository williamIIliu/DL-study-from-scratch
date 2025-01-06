import time
import numpy as np
import pandas as pd
import os

import torch
from torch import nn,optim
from torchinfo import summary
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.dataset import TensorDataset

from config import config
from utils.utils import seed_everything,EarlyStopping
from utils.bi_classifier import performance as bi_performance
from utils.logger import create_logger
from utils.display import loss_and_acc
from models.DET.GradConvEncoderTransformer import GradConvEncoderTransformer

# training plugs
def loss_func(y_true, y_pred):
    loss_fn = nn.BCELoss()
    loss = loss_fn(y_true, y_pred)
    return loss




# model
# from models.DET.GradConvEncoderTransformer import GradConvEncoderTransformer
# det = GradConvEncoderTransformer(input_channel=144,input_length=100,grad=1,dropout=0.25,n_head=8,n_layer=4,d_ff=128,num_out=1,mask=None)
# print(det)
# summary(det, (12, 144, 100))
# import torchvision.models as models
# from torchinfo import summary
# resnet18 = models.resnet18() # 实例化模型
# print(resnet18)
# summary(resnet18, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽

class Trainer(object):
    def __init__(self,model, train_set, val_set, test_set, optimizer,scheduler,epochs,out_path,is_train=True):
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.is_train = is_train
        self.path = out_path

    def train(self):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        if self.is_train == True:
            since = time.time()
            logger.info('------Training logs------')
            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0
                val_running_loss = 0.0
                correct_val = 0
                total_val = 0

                # training phase
                for X_batch, y_batch in self.train_set:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    # feed forward
                    y_hat = self.model(X_batch).to(device)
                    y_batch = y_batch.view(-1, 1)
                    # print("y_hat shape:", y_hat.shape)
                    # print("y_batch shape:", y_batch.shape)
                    loss = loss_func(y_hat, y_batch)

                    # back propagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # loss history
                    running_loss += loss.item()

                    # Calculate training accuracy
                    predicted = (y_hat > 0.5).float()  # Assuming binary classification (sigmoid output)
                    correct_train += (predicted == y_batch).sum().item()
                    total_train += y_batch.size(0)

                self.scheduler.step()
                train_loss = running_loss / len(self.train_set)  # ,optimizer.param_groups[0]['lr']
                train_accuracy = correct_train / total_train
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                # validation phase
                self.model.eval()

                with torch.no_grad():
                    for X_batch, y_batch in self.val_set:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        y_hat = self.model(X_batch)
                        y_batch = y_batch.view(-1, 1)
                        loss = loss_func(y_hat, y_batch)
                        val_running_loss += loss.item()

                        predicted = (y_hat > 0.5).float()
                        correct_val += (predicted == y_batch).sum().item()
                        total_val += y_batch.size(0)

                val_loss = val_running_loss / len(self.val_set)
                val_accuracy = correct_val / total_val
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {running_loss / len(self.train_set):.4f}, "
                    f"Train Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

                # Check Early Stopping
                stopper(val_loss, self.model)
                if stopper.early_stop:
                    logger.info("Early stopping triggered.")
                    break
            time_use = time.time() - since
            logger.info("Total time used: {:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

            details = pd.DataFrame(data={'loss_train': train_losses,
                                         'loss_val': val_losses,
                                         'acc_train': train_accuracies,
                                         'acc_val': val_losses})
            return details
    def test(self):
        self.model.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch_index, (X_batch, y_batch) in enumerate(self.test_set):
                X_batch = X_batch.to(device)
                y_true.extend(y_batch.cpu().numpy())

                output = net(X_batch).cpu().numpy()
                y_pred.extend(output)

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        scores = bi_performance(y_true,y_pred,self.path)

        return scores

if __name__ == '__main__':
    # params
    opts = config.get_options()
    manual_seed = opts.seed
    num_workers = opts.workers
    output_path = opts.output+'/'+opts.model+'/'
    dataset = opts.dataset
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # hyper-parameters
    batch_size = opts.batch_size
    lr = opts.lr
    epochs = opts.epochs

    # logger
    logger = create_logger(output_path)
    logger.info('------Begin Training Model------')
    logger.info(opts)

    # training configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    net = GradConvEncoderTransformer(input_channel=144,
                                     input_length=100,
                                     grad=1,
                                     dropout=0.2,
                                     n_head=2,
                                     n_layer=4,
                                     d_ff=128,
                                     num_out=1,
                                     mask=None).to(device)
    logger.info('------model architecture------')
    logger.info(net)
    # logger.info(summary(net, (2, 144, 100)))

    # optimization
    optimizer = optim.AdamW(net.parameters() , lr=lr,)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    stopper = EarlyStopping(patience=500, delta=0.001, save_path=output_path+f'params_{opts.dataset}.pth')

    # dataset
    trainD_X = np.load(f'../dataset/{dataset}/train_X.npy')
    trainD_Y = np.load(f'../dataset/{dataset}/train_Y.npy')
    testD_X =  np.load(f'../dataset/{dataset}/test_X.npy')
    testD_Y =  np.load(f'../dataset/{dataset}/test_X.npy')
    val_X =    np.load(f'../dataset/{dataset}/val_X.npy')
    val_Y =    np.load(f'../dataset/{dataset}/val_Y.npy')
    # trainD_X = trainD_X
    # val_X = val_X
    # testD_X = testD_X

    X_train = torch.tensor(trainD_X,dtype=torch.float32).to(device)
    Y_train = torch.tensor(trainD_Y,dtype=torch.float32).to(device)
    X_test = torch.tensor(testD_X, dtype=torch.float32).to(device)
    Y_test = torch.tensor(testD_Y, dtype=torch.float32).to(device)
    X_val =   torch.tensor(val_X,dtype=torch.float32).to(device)
    Y_val = torch.tensor(val_Y,dtype=torch.float32).to(device)
    print(X_train.shape, X_test.shape)

    train_dataset = TensorDataset(X_train,Y_train)
    train_iter = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)

    val_dataset = TensorDataset(X_val, Y_val)
    val_iter = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers)

    test_dataset = TensorDataset(X_test, Y_test)
    test_iter = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)
    # print(len(train_iter))
    # print(len(val_iter))
    # print(len(test_iter))

    # train
    trainer = Trainer(net,test_iter,val_iter,test_iter,optimizer,scheduler,epochs,out_path=output_path,is_train=True)
    train_detail = trainer.train()
    loss_train,loss_val,acc_train,acc_val = train_detail['loss_train'], train_detail['loss_val'], train_detail['acc_train'],  train_detail['acc_val']
    loss_and_acc(loss_train,loss_val,acc_train,acc_val,output_path)
    logger.info('------Test performance------')
    test_detail = trainer.test()
    logger.info(test_detail)







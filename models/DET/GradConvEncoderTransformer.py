import torch
from torch import nn,optim
import numpy as np

class ConvEmbedding(nn.Module):
    def __init__(self,SmoothKernel):
        super().__init__()
        #smooth
        self.AvgPool = nn.AvgPool1d(kernel_size=SmoothKernel, stride=1)


        # CNN to extract features
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3, stride=1),#,dilation=2
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3, stride=2),#, dilation=2
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3, stride=1, dilation=2),  # , dilation=2
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten(start_dim=2)




    def forward(self,x):
        x = self.AvgPool(x)
        print('after AVG',x.size())

        x = torch.unsqueeze(x, 1)
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        print('after CNN',x.size())


        # Flatten to project raw signal to grad-shift data shape.
        x = self.flatten(x) #
        print('after flatten',x.size())

        self.fc = nn.Linear(x.size()[2], 128,device=x.device)
        x = self.fc(x)
        self.bn = nn.BatchNorm1d(16,device=x.device)
        x = self.bn(x)
        print('after linear and bn', x.size())
        return x



class Grad4t(nn.Module):
    def __init__(self,num_channel,grad=1):
        super(Grad4t, self).__init__()
        self.grad = grad

        # convolution by each channel
        self.derivConv1d = nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=2,stride=1, bias=False,groups=num_channel)


        # fix weight to do grad operator
        with torch.no_grad():
            kernel = torch.tensor([[-1.0, 1.0]], dtype=torch.float32).repeat(num_channel, 1, 1)
            self.derivConv1d.weight = nn.Parameter(kernel)
    def forward(self, x):
        grad1 = self.derivConv1d(x)
        if self.grad == 2:
            grad2 = self.derivConv1d(grad1)
            return grad1, grad2
        elif self.grad == 1:
            # print(type(grad1))
            # print(grad1.size())
            return grad1
class Grad4x(nn.Module):
    def __init__(self,num_channel,grad=1):
        super(Grad4x, self).__init__()
        self.grad = grad

        # convolution by each channel
        self.derivConv1d = nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=2, stride=1,
                                     bias=False, groups=num_channel)

        # fix weight to do grad operator
        with torch.no_grad():
            kernel = torch.tensor([[-1.0, 1.0]], dtype=torch.float32).repeat(num_channel, 1, 1)
            self.derivConv1d.weight = nn.Parameter(kernel)
    def forward(self, x):
        # transpose for grad t
        x = x.transpose(1,2)


        grad1 = self.derivConv1d(x)
        if self.grad == 2:
            grad2 = self.derivConv1d(grad1)
            grad1 = grad1
            grad2 = grad2
            return grad1.permute(0,2,1), grad2.permute(0,2,1)
        else:
            # print(grad1.size())
            return grad1.permute(0,2,1)




class GradShift(nn.Module):
    def __init__(self,ChannelRaw,LengthRaw,grad):
        super().__init__()
        self.grad = grad
        self.raw_embedding = ConvEmbedding(SmoothKernel=4)
        self.gradx1_embedding = ConvEmbedding(SmoothKernel=4)
        self.gradx2_embedding = ConvEmbedding(SmoothKernel=4)
        self.gradt1_embedding = ConvEmbedding(SmoothKernel=4)
        self.gradt2_embedding = ConvEmbedding(SmoothKernel=4)

        self.grad4t = Grad4t(num_channel=ChannelRaw, grad=grad)
        self.grad4x = Grad4x(num_channel=LengthRaw, grad=grad)

    def forward(self,x):
        raw = self.raw_embedding(x)
        gradx1 = self.grad4t(x) #, gradx2
        # print(gradx1.size())
        gradt1 = self.grad4x(x) #, gradt2

        gradx1 = self.gradx1_embedding(gradx1)
        print(gradx1.shape)
        # gradx2 = self.gradx2_embedding(gradx2)
        gradt1 = self.gradt1_embedding(gradt1)
        # gradt2 = self.gradt2_embedding(gradt2)

        raw =       raw.view(-1,16,128)
        gradx1 = gradx1.view(-1,16,128)
        # gradx2 = gradx2.view(-1,32,128)
        gradt1 = gradt1.view(-1,16,128)
        # gradt2 = gradt2.view(-1,32,128)
        x = torch.cat([raw,gradx1,gradt1,],dim=1) # gradx2,gradt2
        print(x.size())
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model//n_head

        self._W_Q = nn.Linear(d_model,d_model)
        self._W_K = nn.Linear(d_model,d_model)
        self._W_V = nn.Linear(d_model,d_model)

    def ScaledDotProductAttention(self, Q,K,V,attn_mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k) #Q * K.T
        self.attn_scores = attn_scores
        # tmp = attn_scores.cpu().detach().numpy()
        # np.save('../logs/attn_scores.npy',tmp)

        # padding mask
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        # softmax to calculate the weight
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # calculate the V
        output = torch.matmul(attn_probs, V)
        return output

    def split_dmodel_for_heads(self,x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)

    def combine_heads(self,x):
        '''
        在进行复杂的张量操作（如 .transpose()、.permute()）后，
        如果你需要对张量进行 .view() 等需要连续内存的操作时，建议先使用 .contiguous()，以避免潜在的错误或性能问题
        '''
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self,Q, K, V, attn_mask=None):
        # print(self._W_Q(Q).shape)
        Q = self.split_dmodel_for_heads(self._W_Q(Q))
        K = self.split_dmodel_for_heads(self._W_K(K))
        V = self.split_dmodel_for_heads(self._W_V(V))

        attn_output = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        output = self.combine_heads(attn_output)
        return output

class PositionWiseFeedForward(nn.Module):
    '''
    The forward method applies these transformations and activation function sequentially to compute the output.
    This process enables the model to consider the position of input elements while making predictions.
    '''
    def __init__(self, d_model,d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.fc(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model,n_head,d_ff,dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        # attention_weights = attn_output.state_dict()['self_attn.in_proj_weight']

        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class EncoderTransformer(nn.Module):
    def __init__(self,d_model,num_length,n_head,n_layer,d_ff,dropout,num_out,mask=None):
        super(EncoderTransformer, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.mask = mask
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(nn.Linear(num_length*d_model,2048),
                                 # nn.BatchNorm1d(512),
                                 nn.Dropout(dropout),
                                 nn.ReLU(),
                                 nn.Linear(2048,num_out),
                                 nn.Sigmoid())


    def forward(self,x):
        print('x shape in EncoderTransformer',x.shape)
        src_embedded = self.dropout(x)

        features = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(features,self.mask)
        # print('enc_output shape',enc_output.shape)

        batch_size = enc_output.shape[0]
        features_flatten = features.view(batch_size,-1)

        # print('features_flatten shape',features_flatten.shape)
        y_hat = self.mlp(features_flatten) # the dimension of batch_size should be preserved

        # print(y_hat.shape)

        return y_hat




class GradConvEncoderTransformer(nn.Module):
    def __init__(self,input_channel,input_length,grad,dropout, n_head, n_layer, d_ff,num_out=1,mask=None):
        super(GradConvEncoderTransformer, self).__init__()

        self.grad_embedded  =GradShift(input_channel,input_length,grad)

        self.encoder = EncoderTransformer(128,48,n_head,n_layer,d_ff,dropout,num_out=num_out,mask=mask)

    def forward(self,x):
        x_preprocessed = self.grad_embedded(x)
        print('after grad embedded',x_preprocessed.shape)
        out = self.encoder(x_preprocessed)
        return out

if __name__ == '__main__':
    ##
    input_signal = torch.randint(-10, 10, size=(2, 144, 100)).float()

    # raw_layer = ConvEmbedding(144,100,SmoothKernel=4)
    # raw_output = raw_layer(input_signal)
    # print("\nraw embedding:",raw_output.size())
    # # print(raw_output.shape)
    #
    # grad4x_layer = Grad4x(input_signal.size(2), grad=2)
    # grad4x1D_output, grad4x2D_output = grad4x_layer(input_signal)
    # print("\ngrad4t_embedding:")
    # print('1D x grad output\n', grad4x1D_output.size())
    # print('2D x grad output\n', grad4x2D_output.size())
    #
    # grad4t_layer = Grad4t(input_signal.size(1),grad=2)
    # grad4t1D_output,grad4t2D_output = grad4t_layer(input_signal)
    # print("\ngrad4t_embedding:")
    # print('1D t grad output\n',grad4t1D_output.size())
    # print('2D t grad output\n',grad4t2D_output.size())


    #
    # grad_shift = GradShift(144,100,grad=2)
    # print('\ncontact',grad_shift(input_signal).shape)

    # grad_embedded = GradEmbed(input_channel=40,input_length=500,grad=2,dropout=0.25,smooth_k=4,multiscale_k=8)
    # x_grad_preprocess = grad_embedded(input_signal)
    # print(x_grad_preprocess.shape)

    # #####################################################
    # # prepare data
    # #####################################################
    input_signal = torch.randint(-10, 10, size=(2, 144, 100)).float()
    print("true signal:")
    # print(input_signal)

    input_channel = 4,
    input_length = 258,
    grad = 1,
    dropout = 0.25,
    smooth_k = 4,
    multiscale_k = 8
    d_model = 512
    n_head = 8
    n_layer = 6
    d_ff = 2048
    GET = GradConvEncoderTransformer(   input_channel=144,
                                         input_length=100,
                                         grad=1,
                                         dropout=0.25,
                                         n_head=8,
                                         n_layer=4,
                                         d_ff=128,
                                         num_out=1,
                                         mask=None)
    posibility = GET(input_signal)
    print(posibility)
    #
    #
    #

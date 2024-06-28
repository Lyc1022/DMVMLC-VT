import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from loss import Loss
from math import comb
class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        # self.enc_1 = Linear(n_dim, dims[0])
        # self.enc_2 = Linear(dims[0], dims[1])
        # self.enc_3 = Linear(dims[1], dims[2])
        # self.z_layer = Linear(dims[2], n_z)
        # self.z_b0 = nn.BatchNorm1d(n_z)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z


class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Fusion(nn.Module):
    def __init__(self, n_views):
        super(Fusion, self).__init__()
        # 定义可训练参数，用于视图权重
        self.view_weights = nn.Parameter(torch.ones(n_views), requires_grad=True)

    def forward(self, view_representations):
        # 对视图表示加权融合
        weights = torch.nn.functional.softmax(self.view_weights, dim=0)
        weighted_sum = 0
        for view, weight in zip(view_representations, weights):
            weighted_sum += view * weight.unsqueeze(-1)
        return weighted_sum


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, zs):
        x = torch.stack(zs, dim=1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)

        # 在视图维度上对加权值进行求和
        weighted_sum = attended_values.mean(dim=1)

        return weighted_sum


class ViewClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(ViewClassifier, self).__init__()
        self.regression = nn.Linear(input_size, output_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.regression(x)
        x = self.act(x)
        return x


class WeightedSumModel(nn.Module):
    def __init__(self, n_input):
        super(WeightedSumModel, self).__init__()
        self.weights = nn.Parameter(torch.ones(n_input), requires_grad=True)  # 初始化权重为1

    def forward(self, inputs):
        # inputs是一个列表，其中每个元素是一个张量
        weighted_inputs = [w * x for w, x in zip(self.weights, inputs)]
        return torch.stack(weighted_inputs).sum(dim=0)


class AE(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()

        dims = []
        for n_dim in n_input:
            linshi_n_dim = n_dim
            linshidims = []
            for idim in range(n_stacks):
                linshidim = linshi_n_dim
                linshidim = int(linshidim)
                linshidims.append(linshidim)
                linshi_n_dim = round(linshi_n_dim * 1.1)
            linshidims.append(1500)
            linshidims.append(n_z)
            dims.append(linshidims)

        self.autoencoder_list = nn.ModuleList([Autoencoder(encoder_dim=dims[i]) for i in range(len(dims))])

        prediction_dims = [n_z, n_z]
        self.complementors = nn.ModuleList([Missing_Completion(prediction_dims=prediction_dims, views_num=len(n_input) - 1) for _ in range(len(n_input))])

        self.fc = Linear(n_z, n_z)
        self.regression = Linear(n_z, nLabel)
        self.act = nn.Sigmoid()

    def forward(self, mul_X, we):

        individual_zs = []
        x_bar_list = []
        for ae_i, autoencoder in enumerate(self.autoencoder_list):
            x_bar_i, z_i = autoencoder(mul_X[ae_i])
            individual_zs.append(z_i)
            x_bar_list.append(x_bar_i)

        completed_zs = []
        comp_loss = 0
        recon_pre_loss = 0
        for complementor_i, complementor in enumerate(self.complementors):

            completed, per_comp_loss, per_recon_pre_loss = self.complementors[complementor_i](individual_zs, complementor_i, we)

            recon_pre_loss += per_recon_pre_loss

            comp_loss += per_comp_loss
            completed_zs.append(completed)

        comp_loss = comp_loss / len(individual_zs)

        z = torch.stack(completed_zs).mean(dim=0)

        yLable = self.act(self.regression(F.relu(self.fc(F.relu(z)))))

        return x_bar_list, yLable, z, individual_zs, comp_loss, completed_zs, recon_pre_loss


class SharedWeightLayer(nn.Module):
    def __init__(self, num_features):
        super(SharedWeightLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_features, num_features))
        self.bias = nn.Parameter(torch.randn(num_features))

    def forward(self, input_tensor):
        return torch.matmul(input_tensor, self.weight) + self.bias


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim)
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(SharedWeightLayer(self._prediction_dim[i]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth - 1, 0, -1):
            decoder_layers.append(SharedWeightLayer(self._prediction_dim[i]))
            if batchnorm:
                decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class Prediction_Group(nn.Module):
    def __init__(self, prediction_dims):
        super(Prediction_Group, self).__init__()
        self.models = nn.ModuleList([Prediction(prediction_dims) for _ in range(2)])

    def __getitem__(self, index):
        return self.models[index]


class Missing_Completion(nn.Module):
    def __init__(self, prediction_dims, views_num):
        super(Missing_Completion, self).__init__()
        self.predict_model = Prediction(prediction_dim=prediction_dims)

    def forward(self, zs, origin_index, we):
        temp_zs = zs[:]
        temp_list = range(len(zs))
        index_list = [index for index in temp_list if index != origin_index]
        origin_z = temp_zs.pop(origin_index)
        origin_we = we[:, origin_index]
        mse_list = []
        predicted_zs = []
        idx_list = []
        per_comp_loss = 0
        per_recon_pre_loss = 0
        # for model_i, model in enumerate(self.predict_models):
        for model_i in range(len(index_list)):
            # pre_zi = self.predict_models[model_i](temp_zs[model_i])
            pre_zi, recon_pre_zi = self.predict_model(temp_zs[model_i])
            predicted_zs.append(pre_zi)
            temp_we = we[:, index_list[model_i]].bool()
            mask = temp_we & (origin_we.bool())
            common_idx = torch.nonzero(mask, as_tuple=False)
            temp_comp_loss = torch.mean(F.cosine_similarity(origin_z[common_idx], pre_zi[common_idx], dim=1))
            if torch.isnan(temp_comp_loss):
                # 如果是NaN，则将其设置为0或其他合适的值
                temp_comp_loss = 0.0  # 或者其他合适的值

            temp_recon_pre_loss = F.mse_loss(temp_zs[model_i], recon_pre_zi)

            mse_list.append(temp_comp_loss)
            per_comp_loss += temp_comp_loss

            per_recon_pre_loss += temp_recon_pre_loss

            temp_we = (temp_we & (~(origin_we.bool())))
            temp_idx = torch.nonzero(temp_we, as_tuple=False)
            idx_list.append(temp_idx)
            if model_i + 1 == origin_index:
                mse_list.append(1)
                predicted_zs.append(origin_z)
                idx_list.append(torch.nonzero(origin_we, as_tuple=False))

        # sorted_data = sorted(zip(mse_list, predicted_zs, idx_list), key=lambda x: x[0], reverse=True)
        sorted_data = sorted(zip(mse_list, predicted_zs, idx_list), key=lambda x: x[0], reverse=False)
        completed_z = torch.zeros_like(origin_z)
        for mse, pz, idx in sorted_data:
            completed_z[idx] = pz[idx]

        completed_z += 1e-6

        per_comp_loss = per_comp_loss / (len(zs) - 1)
        return completed_z, per_comp_loss, per_recon_pre_loss






class DICNet(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(DICNet, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    def forward(self, mul_X, we):
        x_bar_list, target_pre, fusion_z, individual_zs, comp_loss, completed_zs, recon_pre_loss = self.ae(mul_X, we)
        return x_bar_list, target_pre, fusion_z, individual_zs, comp_loss, completed_zs, recon_pre_loss

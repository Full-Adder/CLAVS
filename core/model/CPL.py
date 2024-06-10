import os
import pytorch_ssim
import cv2
import numpy as np
import torch
import torch.nn as nn
from core.utils.preprocess import reshape_patch_back
from torch.nn import functional as F
from torch.optim import Adam
from core.model.CPL_networks import Prior_FP, shared_encoder, latent_encoder, encoder, Prior, decoder, \
    head_reconstructor
from STA.STAmodel import STANetModel
from STA.loss import KLDLoss
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell as STLSTM


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.STAnet = STANetModel()
        self.configs = configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.shared_encoder = shared_encoder(configs)
        self.latent_encoder = latent_encoder(configs)
        self.decoder = decoder(num_layers, num_hidden, configs)
        self.encoder = encoder(configs)
        self.prior = Prior(configs)
        self.prior_fp = Prior_FP(configs)
        self.head_reconstructor = head_reconstructor(configs)
        self.MSE_criterion = nn.MSELoss()
        self.entropy_criterion = nn.CrossEntropyLoss()
        self.loss2 = nn.BCEWithLogitsLoss()
        self.loss1 = KLDLoss()
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
        self.eps = 1e-8
        self.embed_class = nn.Linear(configs.num_classes, 3 * configs.img_width * configs.img_width)
        self.embed_data = nn.Conv2d(3 * configs.patch_size * configs.patch_size,
                                    3 * configs.patch_size * configs.patch_size, kernel_size=1)

    def reshape_patchback_for_STA(self, patch_tensor, patch_size):
        batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
        img_channels = channels // (patch_size * patch_size)
        a = patch_tensor.contiguous()
        a = a.view([batch_size, seq_length,
                    patch_height, patch_width,
                    patch_size, patch_size,
                    img_channels])
        b = torch.transpose(a, 3, 4).contiguous()
        img_tensor = b.view([batch_size, seq_length,
                             patch_height * patch_size,
                             patch_width * patch_size,
                             img_channels])
        img_tensor = torch.permute(img_tensor, [0,1,4,2,3]).contiguous()
        return img_tensor

    def reshape_patchback(self, image, batch, seq_len, patch_size, height, width):
        first_frame_tensor = image.reshape(batch, seq_len, patch_size, patch_size, 3, height, width)
        fisrt_frame = first_frame_tensor.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        first_frame_patchback = fisrt_frame.reshape(batch, seq_len, 3, height * patch_size, width * patch_size)
        return first_frame_patchback

    def reshape_patch(self, image, batch, patch_size, height, width):
        # [batch, channel, height, width]
        a = image.reshape(batch, 3, height // patch_size, patch_size, width // patch_size, patch_size)
        a = a.permute(0, 3, 5, 1, 2, 4).contiguous()
        image_patch = a.reshape(batch, patch_size * patch_size, 3, height // patch_size, width // patch_size)
        return image_patch

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.configs.batch_size

    def forward(self, frames_tensor, audios, gts, mask_true, category=None, is_train=True, is_replay=False):
        STA_img = self.reshape_patchback_for_STA(frames_tensor, self.configs.patch_size)
        pred_right_num = 0
        # frames_tensor: [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        # audios : [b, l, 81, 199]
        # gts : [b, l, h, w]
        images = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = images.shape[0]
        input_length = images.shape[1]
        height = images.shape[3]
        width = images.shape[4]
        gen_images = []
        cell = []
        hidden = []
        loss_kl = 0
        loss_kl_1 = 0
        loss_kl_2 = 0
        loss_pd = 0
        loss_STA = 0
        loss_cat = torch.tensor(0.0, device=self.configs.device)
        loss_recon = torch.tensor(0.0, device=self.configs.device)
        c_test = []
        h_test = []
        y_gt = torch.eye(self.configs.num_classes, device=self.configs.device)[category]
        y_gt = y_gt.repeat(self.configs.batch_size, 1)
        if is_train:
            for i in range(self.num_layers):
                zeros = torch.zeros(batch, self.num_hidden, height, width).to(self.configs.device)
                cell.append(zeros)
                hidden.append(zeros)
            memory = torch.zeros(batch, self.num_hidden, height, width).to(self.configs.device)
            zeros = torch.zeros(batch, 256).to(self.configs.device)
            h_encode = zeros
            c_encode = zeros
            h_prior = zeros
            c_prior = zeros

        else:
            for i in range(self.num_layers):
                zeros = torch.zeros(batch, self.num_hidden, height, width).to(self.configs.device)
                c_test.append(zeros)
                h_test.append(zeros)
            memory = torch.zeros(batch, self.num_hidden, height, width).to(self.configs.device)
            zeros_ = torch.zeros(batch, 256).to(self.configs.device)
            h_prior_test = zeros_
            c_prior_test = zeros_

        if is_train:

            map0, map0_1, map1, map1_1, map2, map2_1 = self.STAnet(STA_img, audios)
            groundt0, groundt1, groundt2 = gts[:, 0], gts[:, 1], gts[:, 2]
            loss_STA = (self.loss2(map0, groundt0) + self.loss2(map1, groundt1) + self.loss2(map2, groundt2)) \
                         + (self.loss1(F.sigmoid(map0), groundt0) + self.loss1(F.sigmoid(map1), groundt1)
                        + self.loss1(F.sigmoid(map2), groundt2)) \
                         + ((1 - self.ssim_loss(F.sigmoid(map0), groundt0)) + (
                        1 - self.ssim_loss(F.sigmoid(map1), groundt1)) + (
                                    1 - self.ssim_loss(F.sigmoid(map2), groundt2)))

            # img = STA_img[:,0]
            # cv2.imshow("p",np.array(img.to("cpu")[0]))
            # cv2.waitKey(0)
            first_frames_patchback = self.reshape_patchback(images[:, :1], batch, 1, self.configs.patch_size, height,
                                                            width)
            mu_prior, logvar_prior = self.prior_fp(y_gt)
            x_hiddens = self.shared_encoder(first_frames_patchback[:, 0], y_gt)

            mu, logvar = self.latent_encoder(x_hiddens, y_gt)

            loss_kl_1 = self.kl_criterion(mu, logvar, mu_prior, logvar_prior)
            z_recon = self.reparameterize(mu, logvar)
            z_recon = torch.cat([z_recon, y_gt], dim=1)
            gen_first_frame = self.head_reconstructor(z_recon)

            # first frame reconstruction loss
            loss_recon = self.MSE_criterion(first_frames_patchback[:, 0], gen_first_frame)

            input_length = min(self.configs.input_length, input_length)
            for time_step in range(self.configs.total_length - 1):
                if time_step < input_length:
                    input_train = images[:, time_step]
                else:
                    input_train = mask_true[:, time_step - input_length] * images[:, time_step] \
                                  + (1 - mask_true[:, time_step - input_length]) * x_gen

                embedded_class = self.embed_class(y_gt)
                embedded_class = embedded_class.view(batch, -1, height, width)
                embedded_input_target = self.embed_data(images[:, time_step + 1])
                embedded_input_train = self.embed_data(input_train)
                x_target = torch.cat([embedded_input_target, embedded_class], dim=1)
                x_train = torch.cat([embedded_input_train, embedded_class], dim=1)
                mu, logvar, h_encode, c_encode = self.encoder(x_target, h_encode, c_encode, y_gt)
                mu_prior, logvar_prior, h_prior, c_prior = self.prior(
                    x_train, h_prior, c_prior, y_gt)
                loss_kl_2 += self.kl_criterion(mu, logvar, mu_prior, logvar_prior)
                z_train = self.reparameterize(mu, logvar)
                z_train = torch.cat([z_train, y_gt], dim=1)

                hidden, cell, x_gen, memory = self.decoder(
                    input_train, z_train, hidden, cell, memory)

                gen_images.append(x_gen)
                loss_pd += self.MSE_criterion(images[:, time_step + 1], x_gen)
            gen_images = torch.stack(gen_images, dim=0).permute(1, 0, 3, 4, 2).contiguous()


        else:
            pred_right_num = torch.tensor(0)
            if is_replay == False:
                y = y_gt.repeat(self.configs.num_samples, 1)
            else:
                y = y_gt
            input_length = min(self.configs.input_length, input_length)
            for time_step in range(self.configs.total_length - 1):
                if time_step < input_length:
                    input_test = images[:, time_step]
                else:
                    input_test = x_gen_test
                embedded_class = self.embed_class(y)
                embedded_class = embedded_class.view(batch, -1, height, width)
                embedded_input_test = self.embed_data(input_test)
                x_test = torch.cat([embedded_input_test, embedded_class], dim=1)
                mu, logvar, h_prior_test, c_prior_test = self.prior(
                    x_test, h_prior_test, c_prior_test, y)
                z_prior = self.reparameterize(mu, logvar)
                z_prior = torch.cat([z_prior, y], dim=1)
                h_test, c_test, x_gen_test, memory = self.decoder(
                    input_test, z_prior, h_test, c_test, memory)

                gen_images.append(x_gen_test)
            gen_images = torch.stack(gen_images, dim=0).permute(1, 0, 3, 4, 2).contiguous()

            # map0, map0_1, map1, map1_1, map2, map2_1 = self.STAnet(STA_img, audios)

        loss_kl = loss_kl_1 + loss_kl_2
        if loss_kl_2 > 10:
            loss = self.configs.kl_beta * loss_kl + loss_pd + self.configs.total_length * loss_recon + loss_STA
        else:
            loss = loss_pd + self.configs.total_length * loss_recon + 10 * self.configs.kl_beta * loss_kl_1 + loss_STA

        if is_train:
            return gen_images, loss, loss_pd, loss_kl, loss_cat, loss_recon, loss_STA
        else:
            return gen_images, pred_right_num   # todo map0,map1,map2

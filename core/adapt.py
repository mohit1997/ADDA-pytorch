"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

import params
from utils import make_variable


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    src_encoder.train()
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        pBar = tqdm(data_zip)
        c1, s1 = 0, 0
        c2, s2 = 0, 0
        for step, ((images_src, _), (images_tgt, _)) in pBar:
            ###########################
            # 2.1 train discriminator #
            ###########################
            # critic.train()
            # tgt_encoder.eval()
            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0), 1).float())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0), 1).float())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            # print(pred_concat.dtype, label_concat.dtype)
            loss_critic = criterion(pred_concat, label_concat)
            # if (s1 > 1.0) or (s2 < 1.0):
            if (s1 > 0.8) or (s2 < 0.8):
                loss_critic.backward()
                # optimize critic
                optimizer_critic.step()
            
            s1 = ((s1*c1)+(float(loss_critic.item())*len(pred_concat)))/(c1+len(pred_concat))
            c1 += len(pred_concat)

            

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################
            # critic.eval()
            # tgt_encoder.train()
            # zero gradients for optimizer
            for i in range(1):
                optimizer_critic.zero_grad()
                optimizer_tgt.zero_grad()

                # extract and target features
                feat_tgt = tgt_encoder(images_tgt)

                # predict on discriminator
                pred_tgt = critic(feat_tgt)

                # prepare fake labels
                label_tgt = make_variable(torch.ones(feat_tgt.size(0), 1).float())

                # compute loss for target encoder
                loss_tgt = criterion(pred_tgt, label_tgt)
                # if (s1 < 1.0) or (s2 > 5.):
                if True:
                    loss_tgt.backward()

                    # optimize target encoder
                    optimizer_tgt.step()
            s2 = ((s2*c2)+(float(loss_tgt.item())*len(pred_tgt)))/(c2+len(pred_tgt))
            c2 += len(pred_tgt)

            #######################
            # 2.3 print step info #
            #######################
            # if ((step + 1) % params.log_step == 0):
            #     print("Epoch [{}/{}] Step [{}/{}]:"
            #           "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
            #           .format(epoch + 1,
            #                   params.num_epochs,
            #                   step + 1,
            #                   len_data_loader,
            #                   loss_critic.item(),
            #                   loss_tgt.item(),
            #                   acc.item()))
            pBar.set_description("Epoch [{}/{}]:"
                      "d_loss={:.3f} g_loss={:.3f} acc={:.2f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              s1,
                              s2,
                              acc.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder

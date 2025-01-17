"""
Object recognition Things-EEG2 dataset

"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.autograd import Variable
from einops.layers.torch import Rearrange
from Models.VISTA import VISTA, AdaptiveFeatureFusion

gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = '/results/'
model_idx = 'Ours'

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='50', type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = args.lr
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/Data/Things_EEG2/Preprocessed_data_250Hz/'
        self.img_data_path = 'Dnn_feature/'
        self.test_center_path = 'Dnn_feature/'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.model = VISTA()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.fusion = AdaptiveFeatureFusion()
        self.fusion = nn.DataParallel(self.fusion, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy',
                             allow_pickle=True)
        train_data = train_data['preprocessed_eeg_data']
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy',
                            allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_image_train.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_image_test.npy', allow_pickle=True)
        train_label_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_label_train.npy',
                                    allow_pickle=True)
        test_label_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_label_test.npy',
                                   allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)
        train_label_feature = np.squeeze(train_label_feature)
        test_label_feature = np.squeeze(test_label_feature)

        return train_img_feature, test_img_feature, train_label_feature, test_label_feature

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.model = VISTA().cuda()

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        print("Train EEG Size:", train_eeg.shape)
        print("Test EEG Size:", test_eeg.shape)
        print("Test label Size:", test_label.shape)
        train_image_feature, test_image_feature, train_label_feature, test_label_feature = self.get_image_data()
        print("Train Image Size:", train_image_feature.shape)
        print("Test Image Size:", test_image_feature.shape)
        print("Train Label Size:", train_label_feature.shape)
        print("Test Label Size:", test_label_feature.shape)
        test_image_center = test_image_feature
        test_label_center = test_label_feature

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_image_feature = train_image_feature[train_shuffle]
        train_label_feature = train_label_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_image_feature[:740])
        val_label = torch.from_numpy(train_label_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        train_image = torch.from_numpy(train_image_feature[740:])
        train_label = torch.from_numpy(train_label_feature[740:])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image, train_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size,
                                                      shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image, val_label)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size,
                                                          shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        test_image_center = torch.from_numpy(test_image_center)
        test_label_center = torch.from_numpy(test_label_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test,
                                                           shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.fusion.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img, lab) in enumerate(self.dataloader):
                eeg = Variable(eeg.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                lab_features = Variable(lab.cuda().type(self.Tensor))
                fus_features, _ = self.fusion(img_features, lab_features)
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # obtain the features
                [v_eeg_features, s_eeg_features], [loss_time], _, _ = self.model(eeg)
                f_eeg_features, _ = self.fusion(v_eeg_features, s_eeg_features)

                # normalize the features
                f_eeg_features = f_eeg_features / f_eeg_features.norm(dim=1, keepdim=True)
                v_eeg_features = v_eeg_features / v_eeg_features.norm(dim=1, keepdim=True)
                s_eeg_features = s_eeg_features / s_eeg_features.norm(dim=1, keepdim=True)
                fus_features = fus_features / fus_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)
                lab_features = lab_features / lab_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                f_logits_per_eeg = logit_scale * f_eeg_features @ fus_features.t()
                v_logits_per_eeg = logit_scale * v_eeg_features @ img_features.t()
                s_logits_per_eeg = logit_scale * s_eeg_features @ lab_features.t()

                loss_cos_f = self.criterion_cls(f_logits_per_eeg, labels)
                loss_cos_v = self.criterion_cls(v_logits_per_eeg, labels)
                loss_cos_s = self.criterion_cls(s_logits_per_eeg, labels)
                loss = loss_cos_f + loss_cos_v + loss_cos_s + loss_time

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg, vlab) in enumerate(self.val_dataloader):
                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlab_features = Variable(vlab.cuda().type(self.Tensor))
                        vfus_features, _ = self.fusion(vimg_features, vlab_features)
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        [v_veeg_features, s_veeg_features], [vloss_time], _, _ = self.model(veeg)
                        f_veeg_features, _ = self.fusion(v_veeg_features, s_veeg_features)

                        f_veeg_features = f_veeg_features / f_veeg_features.norm(dim=1, keepdim=True)
                        v_veeg_features = v_veeg_features / v_veeg_features.norm(dim=1, keepdim=True)
                        s_veeg_features = s_veeg_features / s_veeg_features.norm(dim=1, keepdim=True)
                        vfus_features = vfus_features / vfus_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)
                        vlab_features = vlab_features / vlab_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        f_vlogits_per_eeg = logit_scale * f_veeg_features @ vfus_features.t()
                        v_vlogits_per_eeg = logit_scale * v_veeg_features @ vimg_features.t()
                        s_vlogits_per_eeg = logit_scale * s_veeg_features @ vlab_features.t()

                        vloss_cos_f = self.criterion_cls(f_vlogits_per_eeg, vlabels)
                        vloss_cos_v = self.criterion_cls(v_vlogits_per_eeg, vlabels)
                        vloss_cos_s = self.criterion_cls(s_vlogits_per_eeg, vlabels)

                        vloss = vloss_cos_f + vloss_cos_v + vloss_cos_s + vloss_time

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            torch.save(self.model.state_dict(),
                                       'trained_model/' + model_idx + '/dependent/model-sub' + str(
                                           self.nSub) + '.pth')
                            torch.save(self.fusion.state_dict(),
                                       'trained_model/' + model_idx + '/dependent/model-sub' + str(
                                           self.nSub) + '_fusion.pth')

                print('Epoch:', e,
                      '  loss fused train: %.4f' % loss_cos_f.detach().cpu().numpy(),
                      '  loss visual train: %.4f' % loss_cos_v.detach().cpu().numpy(),
                      '  loss semantic train: %.4f' % loss_cos_s.detach().cpu().numpy(),
                      '  loss time train: %.4f' % loss_time.detach().cpu().numpy(),
                      '  loss fused val: %.4f' % vloss_cos_f.detach().cpu().numpy(),
                      '  loss visual val: %.4f' % vloss_cos_v.detach().cpu().numpy(),
                      '  loss semantic val: %.4f' % vloss_cos_s.detach().cpu().numpy(),
                      '  loss time val: %.4f' % vloss_time.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: Cos train: %.4f, time train: %.4f, Cos val: %.4f, time val: %.4f\n' % (
                    e, loss_cos.detach().cpu().numpy(), loss_time.detach().cpu().numpy(),
                    vloss_cos.detach().cpu().numpy(), vloss_time.detach().cpu().numpy()))

        # * test part
        all_image_center = test_image_center
        all_label_center = test_label_center
        total = 0
        top1 = [0, 0, 0]
        top3 = [0, 0, 0]
        top5 = [0, 0, 0]

        self.model.load_state_dict(torch.load('trained_model/' + model_idx + '/dependent/model-sub' + str(self.nSub) + '.pth'), strict=False)
        self.fusion.load_state_dict(torch.load('trained_model/' + model_idx + '/dependent/model-sub' + str(self.nSub) + '_fusion.pth'), strict=False)
        self.model.eval()
        self.fusion.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))

                all_image_center = Variable(all_image_center.type(self.Tensor))
                all_label_center = Variable(all_label_center.type(self.Tensor))
                all_fused_center, _ = self.fusion(all_image_center, all_label_center)
                all_image_center = all_image_center / all_image_center.norm(dim=1, keepdim=True)
                all_label_center = all_label_center / all_label_center.norm(dim=1, keepdim=True)
                all_fused_center = all_fused_center / all_fused_center.norm(dim=1, keepdim=True)

                [v_tfea, s_tfea], [tloss_time], _, _ = self.model(teeg)
                f_tfea, _ = self.fusion(v_tfea, s_tfea)
                f_tfea = f_tfea / f_tfea.norm(dim=1, keepdim=True)
                v_tfea = v_tfea / v_tfea.norm(dim=1, keepdim=True)
                s_tfea = s_tfea / s_tfea.norm(dim=1, keepdim=True)
                f_similarity = (100.0 * f_tfea @ all_fused_center.t()).softmax(dim=-1)
                v_similarity = (100.0 * v_tfea @ all_image_center.t()).softmax(dim=-1)
                s_similarity = (100.0 * s_tfea @ all_label_center.t()).softmax(dim=-1)

                # Top-k
                _, f_indices = f_similarity.topk(5)
                _, v_indices = v_similarity.topk(5)
                _, s_indices = s_similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1[0] += (tt_label == f_indices[:, :1]).sum().item()
                top3[0] += (tt_label == f_indices[:, :3]).sum().item()
                top5[0] += (tt_label == f_indices).sum().item()

            top1_acc = [0, 0, 0]
            top3_acc = [0, 0, 0]
            top5_acc = [0, 0, 0]
            top1_acc[0] = float(top1[0]) / float(total)
            top3_acc[0] = float(top3[0]) / float(total)
            top5_acc[0] = float(top5[0]) / float(total)

        print('The test fused Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc[0], top3_acc[0], top5_acc[0]))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test fused Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc[0], top3_acc[0], top5_acc[0]))

        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():
    args = parser.parse_args()

    num_sub = args.num_sub
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    for i in range(num_sub):
        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i + 1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num + 1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + 'result.csv')


if __name__ == "__main__":
    trials = 5
    for f in range(trials):
        print(time.asctime(time.localtime(time.time())))
        main()
        print(time.asctime(time.localtime(time.time())))
        print('-'*100)

from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
import torch.backends.cudnn as cudnn

class TestParams(object):
    # params based on your local env
    gpus = []  # default to use CPU mode

    # loading existing checkpoint
    ckpt = './models/ckpt_epoch_800_res101.pth'     # path to the ckpt file

    testdata_dir = './testimg/'

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params, test_data, train_data):
        assert isinstance(test_params, TestParams)
        self.params = test_params
        self.test_data = test_data
        self.train_data = train_data
        # load model

        self.model = model
        ckpt = self.params.ckpt

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(self.params.gpus) > 0:
            gpu_test = str(self.params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
            if ckpt is not None:
                self._load_ckpt(ckpt)
                logger.info('Load ckpt from {}'.format(ckpt))
        else:
            if ckpt is not None:
                self._load_ckpt(ckpt)
                logger.info('Load ckpt from {}'.format(ckpt))
        self.model.eval()

    def test(self):

        img_list = os.listdir(self.params.testdata_dir)

        for img_name in img_list:
            print('Processing image: ' + img_name)

            img = Image.open(os.path.join(self.params.testdata_dir, img_name))
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)

            print('Prediction number: ' + str(prediction.item()))
    
    def accuracy_for_testdata(self):
        accuracy_sum = 0
        for step, (data, label) in enumerate(self.test_data):
            # train model
            inputs = Variable(data)
            target = Variable(label)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()

            # forward
            score = F.softmax(self.model(inputs), dim=1)
            _,score = torch.max(score.data, dim=1)
            accuracy_sum += sum(score.cpu().data.numpy() == target.cpu().data.numpy())/target.size(0)
            print(target.size(0))

        print(step)
        accuracy = accuracy_sum/(step+1)
        print('The Test Accuracy=%.4f' % accuracy)

    def accuracy_for_traindata(self):
        accuracy_sum = 0
        for step, (data, label) in enumerate(self.train_data):
            # train model
            inputs = Variable(data)
            target = Variable(label)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()

            # forward
            score = F.softmax(self.model(inputs), dim=1)
            _, score = torch.max(score.data, dim=1)
            accuracy_sum += sum(score.cpu().data.numpy() == target.cpu().data.numpy()) / target.size(0)

        accuracy = accuracy_sum / (step+1)
        print('The Train Accuracy=%.4f' % accuracy)
    

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(torch.load(ckpt))

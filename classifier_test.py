from torch import nn
from torch.utils.data import DataLoader
from utils import Tester
from network import resnet34, resnet101, resnet50
from data import Hand
# Set Test parameters
params = Tester.TestParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = r'./models/resnet50/resnet50ckpt_epoch_500.pth' #'./models/ckpt_epoch_800_res101.pth'  #'./models/ckpt_epoch_400_res34.pth'
params.testdata_dir = './data/images/test/signs'
data_root = './data/'
batch_size = 60  # batch_size per GPU, if use GPU mode; resnet34: batch_size=120
num_workers = 2
# loading dadta
print("Loading dataset...")
test_data = Hand(data_root,train=False)
train_data = Hand(data_root, train=True)

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('test dataset len: {}'.format(len(test_dataloader.dataset)))
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

# models
# model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
# model.fc = nn.Linear(512, 6)
# model = resnet101(pretrained=False, num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
# model.fc = nn.Linear(512*4, 6)
model = resnet50(pretrained=False, num_classes=1000)
model.fc = nn.Linear(512*4, 6)
# Test
tester = Tester(model, params, test_dataloader, train_dataloader)
print('test dataset len: {}'.format(len(test_dataloader.dataset)))

tester.accuracy_for_testdata()
tester.accuracy_for_traindata()

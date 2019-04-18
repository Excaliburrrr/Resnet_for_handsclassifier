# Set Training parameters
params = Trainer.TrainParams()
params.max_epoch = 1000
params.criterion = nn.CrossEntropyLoss()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU mode
params.save_dir = model_path
params.ckpt = None
cur_epoch = 0
params.save_freq_epoch = 100

# load data
print("Loading dataset...")
train_data = Hand(data_root,train=True)
val_data = Hand(data_root,train=False)

batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

# models
#model = resnet34(pretrained=False, modelpath=model_path, num_classes=1000)
#batch_size=120, 1GPU Memory < 7000M
#model.fc = nn.Linear(512, 6)
#model = resnet101(pretrained=False, modelpath=model_path, num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
#model.fc = nn.Linear(512*4, 6)
model = resnet50(pretrained=False, modelpath=model_path, num_classes=1000)
model.fc = nn.Linear(512*4, 6)
# optimizer
trainable_vars = [param for param in model.parameters() if param.requires_grad]
print("Training with Adam")
# params.optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
#                                    momentum=momentum,
#                                    weight_decay=weight_decay,
#                                    nesterov=nesterov)
params.optimizer = torch.optim.Adam(trainable_vars, lr=init_lr, betas=(0.9, 0.99), weight_decay=weight_decay)

# Train
params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=lr_decay, patience=10, cooldown=10, verbose=True)
trainer = Trainer(model, params, train_dataloader, cur_epoch, val_dataloader)
trainer.train()

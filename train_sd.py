# Imports
import os
import json
import transforms as T
import torch
from torchvision.transforms import Compose
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from fwi_dataset import FWIDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")


# Model checkpoint paths
resume_paths = {
    "well": "./lightning_logs/curvefault_b_well/checkpoints/epoch=139-step=12039.ckpt",
    "geo": "./lightning_logs/flatfault_b_unconditional/checkpoints/epoch=99-step=48999.ckpt",
    "seis": "./lightning_logs/curvefault_b_seis/checkpoints/epoch=139-step=16099.ckpt",
    "back": "./lightning_logs/curvefault_b_back/checkpoints/epoch=139-step=12039.ckpt"
}

# Load model configuration and initialize
model = create_model('./models/dpm.yaml').cpu()

# Train Conditional Models or Test for Multi-Information Integration 
model.cond_stage_key = ["well"]
model.conditioning_key= "seis_well_back_concat" # change it to the conditon you need: "back", "geo", "well","seis".


print(model.conditioning_key)
assert model.conditioning_key in [
    None, 'seis_well_geo_concat', 'seis_back_concat', 'seis_concat', 'well_concat', 'geo_concat', 
    'back_concat', 'seis_well_concat', 'seis_geo_concat', 'seis_well_back_concat'
]


if_train = False

# Configuration parameters
batch_size = 86
logger_freq = 50000
learning_rate = 1e-4


# Multi-Information Models Loading


if not if_train:
    sufix, model_list = [], []
    model.cond_stage_key = []
    
    for key in ["seis", "well", "back", "geo"]:
        if key in model.conditioning_key:
            sufix.append(f"_{key}")
            model.cond_stage_key.append(key)
            model_list.append(load_state_dict(resume_paths[key], location='cpu'))
    
    # Update model state dictionary
    model_dict = model.state_dict()
    pretrained_dict = {}
    
    for i, sufix_n in enumerate(sufix):
        for k, v in model_list[i].items():
            name = "diffusion_model"
            if name + sufix_n not in k:
                k = k.replace(name, name + sufix_n)
            if k in model_dict:
                pretrained_dict[k] = v
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



# Dataset and annotation paths
anno_path = './split_files'
dataset = 'curvefault-b'
annotations = {
    "train": os.path.join(anno_path, 'curvefault_b_train_test.txt'),
    "val": os.path.join(anno_path, 'curvefault_b_test_test.txt'),
    "test": os.path.join(anno_path, 'curvefault_b_test_test.txt')
}



# Dataset context
file_size = 16
sample_temporal = 1


with open('datasets.json') as f:
    try:
        ctx = json.load(f)[dataset]
    except KeyError:
        print('Unsupported dataset.')
        sys.exit()

if file_size is not None:
    ctx['file_size'] = file_size

# Data transformations
transform_data = Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=1), T.log_transform(ctx['data_max'], k=1))
])
transform_label = Compose([
    T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
])

# Create datasets
train_dataset = FWIDataset(
    annotations["train"], if_test=False, preload=True, lines=-1,
    sample_ratio=sample_temporal, file_size=500,
    transform_data=transform_data, transform_label=transform_label,
    conditioning_key=model.conditioning_key
)

val_dataset = FWIDataset(
    annotations["val"], if_test=True, preload=True, lines=1,
    sample_ratio=sample_temporal, file_size=8,
    transform_data=transform_data, transform_label=transform_label,
    conditioning_key=model.conditioning_key
)

test_dataset = FWIDataset(
    annotations["test"], if_test=True, preload=True, lines=1,
    sample_ratio=sample_temporal, file_size=32,
    transform_data=transform_data, transform_label=transform_label,
    conditioning_key=model.conditioning_key
)

# Data loaders
train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=batch_size, shuffle=False)

# Model Setting
model.learning_rate = learning_rate
trainer = pl.Trainer(
    accelerator="gpu", devices=[1], strategy='ddp', num_nodes=1,
    precision=32, check_val_every_n_epoch=20, max_epochs=150
)

#Model Training
if if_train == True:


  #trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="your_path")
  trainer.fit(model, train_dataloader, val_dataloader)
else:

   trainer.test(model, test_dataloader)
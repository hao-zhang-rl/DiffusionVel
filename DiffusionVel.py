# Imports
import os
import json
import transforms as T
from torchvision.transforms import Compose
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.model import create_model, load_state_dict
from fwi_dataset import FWIDataset
import argparse
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")



parser = argparse.ArgumentParser(description="Accept Parameters from the command")


parser.add_argument(
        "--model_config",
        type=str,
        default='./models/dpm.yaml',
        help="Configuration of Diffusion Models (default: %(default)s)"
    )

parser.add_argument(
    "--conditioning_key",
    type=str,
    choices=['seis_well_geo_concat', 'seis_back_concat', 'seis_concat', 
             'well_concat', 'geo_concat', 'back_concat', 'seis_well_concat', 'seis_geo_concat', 'seis_well_back_concat','well_back_concat'],
    default='seis_concat',
    help="Information type, such as seismic data ('seis'), well log ('well'), background velocity ('back'), or geological information ('geo') (default: %(default)s)"
)
parser.add_argument(
    "--factor_0",
    type=float,
    default=0,
    help="Control factor for the first source"
)
parser.add_argument(
    "--factor_1",
    type=float,
    default=0,
    help="Control factor for the second source"
)
parser.add_argument(
    "--factor_2",
    type=float,
    default=0,
    help="Control factor for the third source"
)
parser.add_argument("--well", type=str, default=None, help="Checkpoint path of well-log GDM")
parser.add_argument("--geo", type=str, default=None, help="Checkpoint path of geological GDM")
parser.add_argument("--seis", type=str, default=None, help="Checkpoint path of seismic GDM")
parser.add_argument("--back", type=str, default=None, help="Checkpoint path of background GDM")

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help=""
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help=""
)
parser.add_argument(
    "--check_val_every_n_epoch",
    type=int,
    default=10,
    help="Validation frequency"
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=200,
    help="Maxium Training Epochs"
)



parser.add_argument(
    "--anno_path",
    type=str,
    default='./split_files',
    help="Directory containing the annotation files for selecting the datasets"
)

parser.add_argument(
    "--dataset",
    type=str,
    default='curvefault-b',
    help="Name of the dataset used in the JSON file"
)

parser.add_argument(
    "--train_anno",
    type=str,
    default='curvefault_b_train_test.txt',
    help="The annotation file for the training dataset (default: %(default)s)"
)
parser.add_argument(
    "--val_anno",
    type=str,
    default='curvefault_b_test_test.txt',
    help="The annotation file for the validation dataset (default: %(default)s)"
)
parser.add_argument(
    "--test_anno",
    type=str,
    default='curvefault_b_test_test.txt',
    help="The annotation file for the testing dataset (default: %(default)s)"
)

parser.add_argument("--if_train", action="store_true", help="Enable training mode")


args = parser.parse_args()




# Construct the resume_paths dictionary
resume_paths = {
    "well": args.well,
    "geo": args.geo,
    "seis": args.seis,
    "back": args.back,
}



if_train = args.if_train

cond_stage_key = []

for key in ["seis", "well", "back", "geo"]:

        if key in args.conditioning_key:
            
            cond_stage_key.append(key)
           

assert args.conditioning_key in [
    None, 'seis_well_geo_concat', 'seis_back_concat', 'seis_concat', 'well_concat', 'geo_concat', 
    'back_concat', 'seis_well_concat', 'seis_geo_concat', 'seis_well_back_concat', 'well_back_concat']



config = OmegaConf.load(args.model_config)
config['model']['params']['conditioning_key'] = args.conditioning_key
config['model']['params']['cond_stage_key'] = cond_stage_key
config['model']['params']['factor_0'] = args.factor_0
config['model']['params']['factor_1'] = args.factor_1
config['model']['params']['factor_2'] = args.factor_2

model = create_model(config).cpu()

# Train Conditional Models or Test for Multi-Information Integration 
model.conditioning_key= args.conditioning_key # change it to the conditon you need: "back", "geo", "well","seis".
    
# Configuration parameters
batch_size = args.batch_size
learning_rate = args.learning_rate


# Multi-Information Models Loading


if not if_train:
    sufix, model_list = [], []
    cond_stage_key = []
    
    for key in ["seis", "well", "back", "geo"]:
        if key in model.conditioning_key:
            sufix.append(f"_{key}")
            cond_stage_key.append(key)
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
else:
    if model.conditioning_key in ['seis_concat', 'well_concat', 'well_concat', 'back_concat']:

        cond_stage_key = model.conditioning_key.replace('_concat', '')


#model.update_key(args.conditioning_key,cond_stage_key)

print(model.cond_stage_key,model.model.conditioning_key)
# Dataset and annotation paths
anno_path = args.anno_path
dataset = args.dataset
annotations = {
    "train": os.path.join(args.anno_path, args.train_anno),
    "val": os.path.join(args.anno_path, args.val_anno),
    "test": os.path.join(args.anno_path, args.test_anno)
}



sample_temporal = 1


with open('datasets.json') as f:
    try:
        ctx = json.load(f)[dataset]
    except KeyError:
        print('Unsupported dataset.')
        sys.exit()

# Data transformations
transform_data = Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=1), T.log_transform(ctx['data_max'], k=1))
])
transform_label = Compose([
    T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
])
if if_train == True:
    # Create datasets
    train_dataset = FWIDataset(
        annotations["train"], if_test=False, preload=True, lines=-0,
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
    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=batch_size, shuffle=False)
else:
    test_dataset = FWIDataset(
        annotations["test"], if_test=True, preload=True, lines=1,
        sample_ratio=sample_temporal, file_size=32,
        transform_data=transform_data, transform_label=transform_label,
        conditioning_key=model.conditioning_key
    )

    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=batch_size, shuffle=False)

# Model Setting
model.learning_rate = learning_rate


trainer = pl.Trainer(
    accelerator="gpu", devices=[1], strategy='ddp', num_nodes=1,
    precision=32, check_val_every_n_epoch=args.check_val_every_n_epoch, max_epochs=args.max_epochs
)

#Model Training
if if_train == True:
  #trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="your_path")
  trainer.fit(model, train_dataloader, val_dataloader)
else:
   model.eval()

 
   trainer.test(model, test_dataloader)
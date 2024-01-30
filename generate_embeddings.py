import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from util.datasets import build_dataset
import os

# call the model
model = models_vit.__dict__['vit_large_patch16'](
    global_pool=True,
)

# load RETFound weights
checkpoint = torch.load('RETFound_cfp_weights.pth', map_location='cuda')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

# assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# manually initialize fc layer
trunc_normal_(model.head.weight, std=2e-5)

# print("Model = %s" % str(model))

model.reset_classifier(num_classes=0)

### Embedding extraction code from here:

# Args class created instead of parsing arguments
class Args:
    def __init__(self, input_size, color_jitter, aa, reprob, remode, recount, data_path) -> None:
        self.input_size = input_size
        self.color_jitter = color_jitter
        self.aa = aa
        self.reprob = reprob
        self.remode = remode
        self.recount = recount
        self.data_path = data_path

args = Args(224, None, 'rand-m9-mstd0.5-inc1', 0.25, 'pixel', 1, './IDRiD_data/')
batch_size = 32


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_embeddings(is_train):
    model.eval()

    split_dir = os.path.join(embed_dir, is_train)
    make_dir(split_dir)

    dataset_val = build_dataset(is_train=is_train, args=args, custom=True)
    classes = dataset_val.classes
    for _class in classes:
        make_dir(os.path.join(split_dir, _class))

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=batch_size,
            num_workers=2,
            drop_last=False
        )

    p = 0
    for x, y, img_path in data_loader_val:
        embeds = model(x)

        for i in range(len(embeds)):
            img_name = img_path[i].split('/')[-1][:-4] + '.jpg'
            embed_path = os.path.join(split_dir, os.path.join(classes[y[i]], img_name))
            torch.save(embeds[i], embed_path)
        p = p + 1
            
        
    
embed_dir = "./idrid_embedding/"   
make_dir(embed_dir)

splits = ['train', 'val', 'test']

for split in splits:
    create_embeddings(split)
import torch, json
from main import build_model_main
from util.slconfig import SLConfig
from PIL import Image
import datasets.transforms as T
import os

model_config_path = "config/DINO/DINO_4scale_swin.py"  # change the path of the model config file
model_checkpoint_path = "/media/palm/BiggerData/superai/baseline_reality/liver/cp/DINO_swinL/checkpoint0011.pth"  # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
args.num_classes = 4
args.backbone_dir = 'backbones'

args.amp = True
args.dn_label_coef = 1.0
args.dn_bbox_coef = 1.0
args.dn_scalar = 100
args.embed_init_tgt = True
args.use_ema = False
args.dn_box_noise_scale = 1.0

model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model = model.eval()

# load coco names
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k): v for k, v in id2name.items()}

args.dataset_file = 'coco'
args.coco_path = "/comp_robot/cv_public_dataset/COCO2017/"  # the path of coco
args.fix_size = False

# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# visualize outputs
thershold = 0.3  # set a thershold
print()

path = '/media/palm/BiggerData/superai/baseline_reality/liver/new_data/distribute/test/images'
annotations = {}
cats = ['bg', "cystic", "FFS", "solid"]
for filename in os.listdir(path):
    image = Image.open(os.path.join(path, filename)).convert("RGB")  # load image
    w = image.width
    h = image.height
    image, _ = transform(image, None)
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.tensor([[1.0, 1.0]]).cuda())[0]
    annotation = []
    for i in range(output['scores'].size(0)):
        # if output['scores'][i] < 0.3:
        #     break
        x1, y1, x2, y2 = output['boxes'][i].cpu().numpy().tolist()
        label = output['labels'][i].cpu().numpy().tolist()
        score = output['scores'][i].cpu().numpy().tolist()
        annotation.append([x1 * w, y1 * h, x2 * w, y2 * h, score, cats[label]])
    annotations[filename] = annotation
json.dump(annotations, open(f'results/liver/swin_L_4scale_all.json', 'w'))

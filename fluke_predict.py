import torch, json
from main import build_model_main
from util.slconfig import SLConfig
from PIL import Image
import datasets.transforms as T
import os

model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
model_checkpoint_path = "/media/palm/BiggerData/Chula_Parasite/new/checkpoints/DINO_R50_4/checkpoint_best_regular.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
args.num_classes = 13
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# load coco names
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}

args.dataset_file = 'coco'
args.coco_path = "/comp_robot/cv_public_dataset/COCO2017/" # the path of coco
args.fix_size = False


# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# visualize outputs
thershold = 0.3 # set a thershold
print()

data = json.load(open('/home/palm/PycharmProjects/chula_fluke/jsns/val.json'))
annotations = {}
for file in data:
    if file['filename'].startswith('val') or file['filename'].startswith('test'):
        d = '/media/palm/data/MicroAlgae/22_11_2020/'
        if not os.path.exists(os.path.join(d, file['filename'])):
            file['filename'] = file['filename'].replace('val', 'test')
    else:
        d = '/media/palm/BiggerData/Chula_Parasite/Chula-ParasiteEgg-11/Chula-ParasiteEgg-11/'
    image = Image.open(os.path.join(d, file['filename'])).convert("RGB")  # load image
    w = image.width
    h = image.height
    image, _ = transform(image, None)
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.tensor([[1.0, 1.0]]).cuda())[0]
    annotation = []
    for i in range(output['scores'].size(0)):
        if output['scores'][i] < 0.3:
            break
        x1, y1, x2, y2 = output['boxes'][i].cpu().numpy().tolist()
        label = output['labels'][i].cpu().numpy().tolist()
        score = output['scores'][i].cpu().numpy().tolist()
        annotation.append([x1*w, y1*h, x2*w, y2*h, label, score])
    annotations[file['filename']] = annotation
json.dump(annotations, open(f'results/fluke/R50_4scalre.json', 'w'))

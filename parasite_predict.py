import torch, json
from main import build_model_main
from util.slconfig import SLConfig
from PIL import Image
import datasets.transforms as T
import os

model_config_path = "config/DINO/DINO_4scale_swin.py"  # change the path of the model config file
model_checkpoint_path = "/media/palm/BiggerData/parasites/cp/DINO_Swin-L/checkpoint0011.pth"  # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
args.num_classes = 8
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
CLASSES = ('bg', 'al', 'cs', 'ec', 'mif', 'ov', 'tspp', 'tt')

data = json.load(open('/home/palm/PycharmProjects/parasites/annotations/val.json'))
annotations = {}
d = '/media/palm/BiggerData/parasites/parasites'
for file in data:
    image = Image.open(os.path.join(d, file['filename'])).convert("RGB")  # load image
    w = image.width
    h = image.height
    image, _ = transform(image, None)
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.tensor([[1.0, 1.0]]).cuda())[0]
    annotation = {
        'file_name': file['filename'],
        'bboxes': []
    }
    for i in range(output['scores'].size(0)):
        if output['scores'][i] < 0.3:
            break
        x1, y1, x2, y2 = output['boxes'][i].cpu().numpy().tolist()
        label = output['labels'][i].cpu().numpy().tolist()
        score = output['scores'][i].cpu().numpy().tolist()
        class_name = CLASSES[label]
        annotation['bboxes'].append([x1*w, y1*h, x2*w, y2*h, class_name])
    annotations[file['filename']] = annotation
json.dump(annotations, open(f'/home/palm/PycharmProjects/parasites/results/prediction/DINO_Swin-L.json', 'w'))

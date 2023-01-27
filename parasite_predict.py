import torch, json
from main import build_model_main
from util.slconfig import SLConfig
from PIL import Image
import datasets.transforms as T
import os
import cv2
import numpy as np

colors = {
    'al': (120, 190, 100),
    'cs': (255, 40, 80),
    'ec': (250, 200, 9),
    'mif': (60, 250, 10),
    'ov': (200, 80, 200),
    'tspp': (0, 200, 200),
    'tt': (160, 255, 255),
}


def add_bbox(img, bbox, labels, color, conf=None, show_txt=True, pos='top'):
    # bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    if conf:
        txt = '{}:{:.1f}'.format(labels, conf)
    else:
        txt = labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    if show_txt:
        if pos == 'top':
            y1 = bbox[1] - cat_size[1] - 2
            y2 = bbox[1]
        else:
            y1 = bbox[3]
            y2 = bbox[3] + cat_size[1]
        cv2.rectangle(img,
                      (bbox[0], y1),
                      (bbox[0] + cat_size[0], y2), color, -1)
        cv2.putText(img, txt, (bbox[0], y2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img


model_config_path = "config/DINO/DINO_4scale_swin.py"  # change the path of the model config file
model_checkpoint_path = "/media/palm/BiggerData/parasites/cp/DINO_fixed/checkpoint0011.pth"  # change the path of the model checkpoint
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
outputs = {}
for c in CLASSES:
    outputs[c] = []

data = json.load(open('/home/palm/PycharmProjects/parasites/annotations/val_al_removed.json'))
annotations = {}
d = '/media/palm/BiggerData/parasites/parasites'
for file in data:
    c = file['filename'].split(' - ')[0]
    if len(outputs[c]) > 5:
        continue
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
    image = cv2.imread(os.path.join(d, file['filename']))
    for i in range(len(file['ann']['bboxes'])):
        bbox = file['ann']['bboxes'][i]
        class_name = CLASSES[file['ann']['labels'][i]]
        image = add_bbox(image, bbox, class_name, (0, 0, 255), show_txt=False)

    for i in range(output['scores'].size(0)):
        if output['scores'][i] < 0.3:
            break
        x1, y1, x2, y2 = output['boxes'][i].cpu().numpy().tolist()
        label = output['labels'][i].cpu().numpy().tolist()
        score = output['scores'][i].cpu().numpy().tolist()
        class_name = CLASSES[label]
        annotation['bboxes'].append([x1*w, y1*h, x2*w, y2*h, class_name])
        bbox = np.array([x1*w, y1*h, x2*w, y2*h]).astype('int')
        image = add_bbox(image, bbox, class_name, (colors[class_name]))

    outputs[c].append(annotation)
    annotations[file['filename']] = annotation
    cv2.imwrite(os.path.join('/media/palm/BiggerData/parasites/out_al_removed', c + '_' + os.path.basename(annotation['file_name'])), image)
# json.dump(annotations, open(f'/home/palm/PycharmProjects/parasites/results/prediction/DINO_Swin-L.json', 'w'))

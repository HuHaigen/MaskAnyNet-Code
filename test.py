import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import valDataSet
from models.GuideResNet import GuideResNet34 as create_model
# from models.EfficientNetV2 import efficientnetv2_s as create_model
# from models.GuideSwin import swin_tiny_patch4_window7_224 as create_model
# from models.GuideVit import vit_base_patch16_224_in21k as create_model
from utils.utils_Vit import read_test_data, test


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    test_images_path, test_images_label = read_test_data(args.data_path)

    img_size = 224
    data_transform = {
        "test": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   #transforms.Resize(int(img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test_anthor": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   #transforms.Resize(int(img_size // 2)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    test_dataset = valDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform1=data_transform["test"],
                            transform2=data_transform["test_anthor"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # test
    test_top1_acc, test_top5_acc = test(model=model, data_loader=test_loader, device=device)
    print("Top1_acc: {:.4f}".format(test_top1_acc) + "\n" "Top5_acc: {:.4f}".format(test_top5_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2)

    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        # default="D:/data_resources/classfication/CIFAR-100/test")
                        default="D:/data_resources/classfication/tiny-ImageNet-200/val")

    parser.add_argument('--weights', type=str,
                        # default='/export/home/hjs/project/deep-learning-for-image-processing-master/pytorch_classification/GuideNet/weights/best_model288.pth')
                        default = './weights/best_model162.pth')
                        # default='./weights/Cutout/efficientV2/Cifar-100/78.20_94.29.pth')
                        # D:\pycharmproject\deep-learning-for-image-processing-master\pytorch_classification\GuideNet\weights\GridMask\Swin\no-pretrain\Ciafr-10\Swinmodel-103.pth
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
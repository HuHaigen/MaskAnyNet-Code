import os
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Replace the following import with your actual model definition
# from models.EfficientNetV2 import efficientnetv2_s as create_model
from models.GuideResNet import GuideResNet34 as create_model


class GradCAM:
    """
    A simple Grad-CAM implementation:
      - Registers forward and backward hooks on a target layer
      - Stores feature maps (activations) in forward hook
      - Stores gradients in backward hook
      - Combines activations and gradients to generate CAM heatmap
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.forward_hook_handle = self.target_layer.register_forward_hook(self.forward_hook)
        self.backward_hook_handle = self.target_layer.register_backward_hook(self.backward_hook)

        self.activations = None
        self.gradients = None

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove_hooks(self):
        """Remove hooks to prevent duplication"""
        self.forward_hook_handle.remove()
        self.backward_hook_handle.remove()

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, class_idx: int = None):
        """
        Forward and backward pass to compute Grad-CAM heatmap
        :param x1: input of main branch
        :param x2: input of auxiliary branch
        :param class_idx: target class to visualize (default: predicted class)
        :return: CAM heatmap (same size as input x1)
        """
        logits = self.model(x1, x2)  # [batch_size, num_classes]
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        score = logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients
        b, c, h, w = gradients.shape

        alpha = gradients.view(b, c, -1).mean(2)  # [b, c]
        weights = alpha[:, :, None, None]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [b, 1, h, w]
        cam = nn.functional.relu(cam)

        cam = nn.functional.interpolate(cam, size=(x1.shape[-2], x1.shape[-1]), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.squeeze(1)

        return cam


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, alpha: float = 0.85):
    """
    Overlay CAM heatmap on the original image
    :param img: original image as numpy array, shape=(H, W, 3), range [0,1] or [0,255]
    :param mask: CAM heatmap, shape=(H, W), range [0,1]
    :param alpha: transparency ratio for heatmap overlay
    :return: overlay image as numpy array
    """
    if img.max() > 1:
        img = img.astype(np.float32) / 255.

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0

    overlay = heatmap * alpha + np.float32(img)
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 200
    img_size = 224

    data_transform = transforms.Compose([
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transform2 = transforms.Compose([
        transforms.Resize((int(img_size)//2, int(img_size)//2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_path = "C:/Users/Juma_Luata/Desktop/test/test_2963.jpeg"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."

    img_pil = Image.open(img_path).convert('RGB')
    resized_img_pil = img_pil.resize((img_size, img_size))
    original_image = np.array(resized_img_pil)

    x1 = data_transform(img_pil)
    x2 = data_transform2(img_pil)
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)

    json_path = './cifar-10-class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "./weights/Grid+Cut+Random/ResNet/Tiny-ImageNet/0.1+0.2+0.1/62.07_81.58.pth"
    assert os.path.exists(model_weight_path), f"file: '{model_weight_path}' does not exist."
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(x1.to(device), x2.to(device))
        output = torch.squeeze(output).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1].conv2
    else:
        target_layer = model.conv2  # adjust accordingly

    grad_cam = GradCAM(model=model, target_layer=target_layer)
    cam_mask = grad_cam(x1.to(device), x2.to(device), class_idx=predict_cla)
    grad_cam.remove_hooks()
    cam_mask_np = cam_mask[0].detach().cpu().numpy()
    result_cam = show_cam_on_image(original_image, cam_mask_np)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Resized Input Image")
    plt.imshow(original_image)

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(result_cam[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    main()

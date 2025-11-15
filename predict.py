import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from models.EfficientNetV2 import efficientnetv2_s as create_model
from models.GuideResNet import GuideResNet34 as create_model

def visualize_and_save_features(features, layer_names=None, original_image=None, heat=False):
    for i, feature_map in enumerate(features):
        if layer_names is not None:
            layer_name = layer_names[i]
            path = os.path.join("Visualize_Features", layer_name)
            os.makedirs(path, exist_ok=True)
            print(f"Visualizing layer: {layer_name}")

        for j in range(feature_map.shape[1]):  # Iterate through channels
            if feature_map.shape[2] > 0 and feature_map.shape[3] > 0:  # Check height and width
                # Save individual feature maps
                plt.imshow(feature_map[0, j].cpu().detach().numpy(), cmap='viridis')  # Use first sample
                plt.title(f"{layer_name} - Feature {j}")
                plt.colorbar()
                plt.savefig(os.path.join(path, f"feature_{j}.png"))
                plt.close()

                # Overlay heatmap on original image
                if original_image is not None and heat:
                    heatmap = feature_map[0, j].cpu().detach().numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize

                    plt.imshow(original_image)
                    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
                    plt.title(f"{layer_name} - Heatmap {j}")
                    plt.axis('off')
                    plt.savefig(os.path.join(path, f"heatmap_{j}.png"))
                    plt.close()
            else:
                print(f"Feature map {j} has invalid shape: {feature_map.shape}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 100
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transform2 = transforms.Compose([
        transforms.Resize(int(img_size)//2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load image
    img_path = "D:/data_resources/classfication/CIFAR-100/test/bicycle/bicycle_s_001102.png"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path)
    img2 = img.copy()
    plt.imshow(img)

    # Save original image for heatmap visualization
    original_image = img.copy()

    # Apply transformations
    img = data_transform(img)
    img2 = data_transform2(img2)

    # Expand dimensions to [N, C, H, W]
    img = torch.unsqueeze(img, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)

    # Load class index to label mapping
    json_path = './cifar-100-class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    # Create model instance
    model = create_model(num_classes=num_classes).to(device)
    Vis = True

    # Load model weights
    model_weight_path = "./weights/Grid+Cut+Random/ResNet/Cifar-100/0.1+0.3+0.0/add/74.69_92.30.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # Perform forward pass and get features
        output, features = model(img.to(device), img2.to(device), return_features=True)
        output = torch.squeeze(output).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = f"class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla].numpy():.4}"
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.4}".format(class_indict[str(i)], predict[i].numpy()))
    plt.show()

    # Visualize feature maps
    if Vis:
        model_name = create_model.__name__
        if "efficientnetv2" in model_name:
            layer_names = ["stem1", "stem2", "Fusion"] + [f"block_{i + 1}" for i in range(len(model.blocks))]
        else:  # ResNet
            layer_names = ["Path1", "Path2", "Fusion", "Layer_1", "Layer_2", "Layer_3", "Layer_4"]
        visualize_and_save_features(features, layer_names, original_image=original_image, heat=False)

if __name__ == '__main__':
    main()

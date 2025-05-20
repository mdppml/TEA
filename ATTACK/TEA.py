import torch.nn.functional as F
import random
import requests, json
import time
import torchvision.transforms as transforms
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import copy
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop
import os
import torchvision.models as torch_models
torch_available = False
try:
    import torch
    from PIL import Image

    torch_available = True
except ImportError:
    from PIL import Image

transform = Compose([Resize(256), CenterCrop(224)])
if torch_available:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'

def inv_tf(x, mean, std):
    for i in range(len(mean)):
        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)
    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)
    return x

def is_adversarial(image, model, tar_lbl):
    predict_label = torch.argmax(model.forward(Variable(image, requires_grad=True)).data).item()
    is_adv = predict_label == tar_lbl
    return 1 if is_adv else -1

def create_gaussian_weight(patch_height, patch_width, sigma=None, device='cpu'):
    if sigma is None:
        sigma = min(patch_height, patch_width) / 2.0
    y = torch.arange(patch_height, device=device).float() - (patch_height - 1) / 2.0
    x = torch.arange(patch_width, device=device).float() - (patch_width - 1) / 2.0
    y = y.view(-1, 1)
    x = x.view(1, -1)
    gauss = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.max()
    return gauss.unsqueeze(0).unsqueeze(0)

def extract_edge_data(image_tensor, low_threshold=50, high_threshold=150):
    image_np = image_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np, 0, 1)
    gray_image = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140])
    sx = ndimage.sobel(gray_image, axis=0, mode='reflect')
    sy = ndimage.sobel(gray_image, axis=1, mode='reflect')
    grad_magnitude = np.hypot(sx, sy)
    grad_magnitude = (grad_magnitude / (grad_magnitude.max() + 1e-8)) * 255
    grad_magnitude = grad_magnitude.astype(np.uint8)
    edge_mask = np.zeros_like(grad_magnitude)
    edge_mask[(grad_magnitude >= low_threshold) & (grad_magnitude <= high_threshold)] = 255
    return edge_mask, gray_image

def create_soft_edge_mask(edge_mask, blur_size=5, intensity=0.8):
    soft_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (blur_size, blur_size), 0)
    soft_mask = soft_mask / soft_mask.max()
    soft_mask = (soft_mask * intensity).astype(np.float32)
    return torch.tensor(soft_mask, dtype=torch.float32)

def denormalize_image(image_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    denorm_image = image_tensor.detach().cpu() * std + mean
    denorm_image = torch.clamp(denorm_image, 0, 1)
    return denorm_image.squeeze(0).permute(1, 2, 0).numpy()

def show_image_with_label(img_tensor, src_img, mean, std, total_calls, title=None, src_lbl=None, tar_lbl=None,
                          is_adv=None, caption=None):
    img_np = denormalize_image(img_tensor, mean, std)
    src_np = denormalize_image(src_img, mean, std)
    dist = np.linalg.norm(img_np - src_np)
    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.axis("off")
    if is_adv == 1:
        lbl = tar_lbl
    else:
        lbl = src_lbl
    fig.text(0.5, 0.065, f"Label = {lbl}", ha='center', va='top', fontsize=12)
    if total_calls != 0:
        fig.text(0.5, -0.055, f"Total queries used = {total_calls}", ha='center', va='top', fontsize=10)
    if title:
        fig.suptitle(title, fontsize=14)
    if title != "Source":
        if caption:
            fig.text(0.5, 0.005, f"ℓ² to source = {caption:.4f}", ha='center', va='top', fontsize=10)
        else:
            fig.text(0.5, 0.005, f"ℓ² to source = {dist:.4f}", ha='center', va='top', fontsize=10)
    plt.show()

def preprocess_image(image, mean, std, device):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
    return transform(image).to(device)

def global_search(x_0, x_random, soft_edge_mask_y, model, tar_lbl, initial_step_factor=0.0001, momentum=0.9, max_calls=20):
    num_calls = 0
    direction = x_0 - x_random
    velocity = torch.zeros_like(direction)
    current_point = x_random.clone()
    last_adv_point = current_point.clone()
    initial_distance = torch.norm(direction).item()
    step_size = initial_distance * initial_step_factor
    non_edge_mask_y = (1 - soft_edge_mask_y)
    while num_calls < max_calls:
        velocity = momentum * velocity + (1 - momentum) * direction
        step = step_size * velocity
        non_edge_mask_y = non_edge_mask_y.to(current_point.device)
        next_point = current_point + step * non_edge_mask_y
        num_calls += 1
        if is_adversarial(next_point, model, tar_lbl) == 1:
            last_adv_point = next_point.clone()
            current_point = next_point
            step_size *= 1.1
        else:
            step_size *= 0.5
            break
        if torch.norm(step).cpu().numpy() < 0.0001:
            break
    return last_adv_point, num_calls

def patch_search(x_0, x_random, soft_edge_mask_y, model, tar_lbl, mean, std, min_patch_size=16, max_patch_size=64, step_size_factor=0.005, momentum=0.9, max_calls=20, iterations_to_show=3):
    total_calls = 0
    best_x_random = x_random.clone()
    _, _, H, W = x_random.shape
    patch_boundary_mask = torch.zeros((H, W), device=x_random.device, dtype=torch.float32)
    x0_inv = inv_tf(copy.deepcopy(x_0.cpu()[0].squeeze()), mean, std)
    overall_break = False
    last_query = 0
    rarity_break = 0
    recorded = []
    record_count = 0
    while True:
        diff_map = torch.abs(x_0 - best_x_random).mean(dim=1, keepdim=True)
        diff_map = F.avg_pool2d(diff_map, kernel_size=3, stride=1, padding=1)
        diff_flat = diff_map.view(-1)
        high_diff_indices = torch.argsort(diff_flat, descending=True)[:H * W // 5]
        center_idx = high_diff_indices[random.randint(0, len(high_diff_indices) - 1)]
        i_center = (center_idx // W).item()
        j_center = (center_idx % W).item()
        patch_size = random.randint(min_patch_size, max_patch_size)
        i_start = max(0, i_center - patch_size // 2)
        j_start = max(0, j_center - patch_size // 2)
        i_end = min(H, i_start + patch_size)
        j_end = min(W, j_start + patch_size)
        bw = 4
        patch_boundary_mask[i_start:min(i_start + bw, H), j_start:j_end] = 1
        patch_boundary_mask[max(i_end - bw, 0):i_end, j_start:j_end] = 1
        patch_boundary_mask[i_start:i_end, j_start:min(j_start + bw, W)] = 1
        patch_boundary_mask[i_start:i_end, max(j_end - bw, 0):j_end] = 1
        pre_patch = best_x_random.clone()
        pre_patch_inv = inv_tf(copy.deepcopy(pre_patch.cpu()[0].squeeze()), mean, std)
        pre_norm = torch.norm(x0_inv - pre_patch_inv).item()
        patch_updated = False
        for iteration in range(max_calls):
            if total_calls > 25 + last_query or rarity_break > 5000:
                overall_break = True
                break
            local_direction = x_0[:, :, i_start:i_end, j_start:j_end] - best_x_random[:, :, i_start:i_end,j_start:j_end]
            momentum_patch = momentum * torch.zeros_like(local_direction) + (1 - momentum) * local_direction
            step_size = torch.norm(x_0 - best_x_random).item() * step_size_factor
            patch_h, patch_w = i_end - i_start, j_end - j_start
            gaussian_weight = create_gaussian_weight(patch_h, patch_w, sigma=patch_size / 4.0, device=x_random.device)
            mask_patch = soft_edge_mask_y[:, :, i_start:i_end, j_start:j_end].to(gaussian_weight.device)
            update_weight = gaussian_weight * (1 - 0.9 * mask_patch)
            step = step_size * momentum_patch * update_weight
            next_patch = best_x_random[:, :, i_start:i_end, j_start:j_end] + step
            temp_x_random = best_x_random.clone()
            temp_x_random[:, :, i_start:i_end, j_start:j_end] = next_patch
            new_distance = torch.norm(x_0 - temp_x_random).item()
            if new_distance >= 0.999 * torch.norm(x_0 - best_x_random).item():
                rarity_break += 1
                break
            total_calls += 1
            if is_adversarial(temp_x_random, model, tar_lbl) == 1:
                patch_updated = True
                rarity_break = 0
                best_x_random[:, :, i_start:i_end, j_start:j_end] = next_patch.clone()
                last_query = total_calls
            else:
                break
            if overall_break:
                break
        if patch_updated and record_count < iterations_to_show:
            post_patch_inv = inv_tf(copy.deepcopy(best_x_random.cpu()[0].squeeze()), mean, std)
            post_norm = torch.norm(x0_inv - post_patch_inv).item()
            before_np = denormalize_image(pre_patch, mean, std)
            after_np = denormalize_image(best_x_random, mean, std)
            mask_map = torch.zeros((H, W), dtype=torch.bool)
            mask_map[i_start:i_end, j_start:j_end] = True
            highlight = before_np.copy()
            highlight[~mask_map.numpy()] = (highlight[~mask_map.numpy()] * 0.5).astype(highlight.dtype)
            recorded.append((before_np, highlight, after_np, pre_norm, post_norm))
            record_count += 1
        if overall_break:
            break
    if recorded:
        rows = len(recorded)
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        for i, (bef, hl, aft, pre_d, post_d) in enumerate(recorded):
            row_axes = axes[i] if rows > 1 else axes
            for j, img in enumerate((bef, hl, aft)):
                ax = row_axes[j]
                ax.imshow(img)
                ax.axis('off')
                if i == 0:
                    ax.set_title(("Before", "Selected Patch", "After")[j])
                if j == 2:
                    ax.text(
                        0.5, -0.025,
                        f"ℓ² to source  = {post_d:.4f}",
                        ha='center', va='top',
                        transform=ax.transAxes,
                        fontsize=10
                    )
        plt.tight_layout()
        plt.show()
    return best_x_random, total_calls, patch_boundary_mask

def initialize_attack(pair_id, src_img, tar_img, model_arch='ViT'):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_arch == 'resnet50':
        net = torch_models.resnet50(pretrained=True)
    elif model_arch == 'resnet101':
        net = torch_models.resnet101(pretrained=True)
    elif model_arch == 'vgg16':
        net = torch_models.vgg16(pretrained=True)
    elif model_arch == 'ViT':
        import timm
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")
    net = net.to(device)
    net.eval()
    model = net
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x_0 = preprocess_image(src_img, mean, std, device).unsqueeze(0)
    x_t = preprocess_image(tar_img, mean, std, device).unsqueeze(0)
    orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()
    tar_label = torch.argmax(net.forward(Variable(x_t, requires_grad=True)).data).item()
    if orig_label == tar_label:
        print(f"Pair {pair_id}: Initial and target images belong to the same class. Skipping attack.")
        return
    src_img = x_0
    tar_img = x_t
    src_lbl = torch.argmax(model.forward(src_img)).item()
    tar_lbl = None
    if tar_img is not None:
        tar_lbl = torch.argmax(model.forward(tar_img)).item()
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    response = requests.get(url)
    idx_to_label = {int(k): v[1] for k, v in json.loads(response.content).items()}
    src_lbl_name = idx_to_label[src_lbl]
    tar_lbl_name = idx_to_label[tar_lbl]
    src_np = denormalize_image(src_img, mean, std)
    tar_np = denormalize_image(tar_img, mean, std)
    dist = np.linalg.norm(tar_np - src_np)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(src_np)
    axes[0].axis('off')
    axes[0].set_title(f"Source\nLabel = {src_lbl_name}", fontsize=12)
    axes[1].imshow(tar_np)
    axes[1].axis('off')
    axes[1].set_title(f"Target\nLabel = {tar_lbl_name}", fontsize=12)
    axes[1].text(
        0.5, -0.025,
        f"ℓ² to source = {dist:.4f}",
        ha='center', va='top',
        transform=axes[1].transAxes,
        fontsize=10
    )
    plt.suptitle(f"Pair {pair_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return_arguments = [model, mean, std, src_img, tar_img, tar_lbl, src_lbl_name, tar_lbl_name]
    return return_arguments

def edge_mask_initialization(mean, std, src_img, tar_img):
    x_inv = inv_tf(copy.deepcopy(src_img.cpu()[0, :, :, :].squeeze()), mean, std)
    x_adv = tar_img
    x_adv_inv = inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()), mean, std)
    norm = torch.norm(x_inv - x_adv_inv)
    edge_mask_adv, gray_adv = extract_edge_data(x_adv)
    soft_edge_mask_adv = create_soft_edge_mask(edge_mask_adv)
    orig_np = denormalize_image(x_adv, mean, std)
    edge_np = edge_mask_adv
    soft_np = soft_edge_mask_adv.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_np)
    axes[0].axis('off')
    axes[0].set_title('Target')
    axes[1].imshow(edge_np, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Edge Mask')
    axes[2].imshow(soft_np, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Soft Edge Mask')
    plt.tight_layout()
    plt.show()
    return soft_edge_mask_adv

def global_edge_informed_search(soft_edge_mask_adv, model, mean, std, src_img, tar_img, tar_lbl, src_lbl_name,tar_lbl_name, iterations_to_show=3):
    total_calls = 0
    x_inv = inv_tf(copy.deepcopy(src_img.cpu()[0, :, :, :].squeeze()), mean, std)
    x_adv = tar_img
    x_adv_inv = inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()), mean, std)
    norm = torch.norm(x_inv - x_adv_inv)
    previous_norm = norm
    continuous = 0
    _ = 0
    recorded = []
    for i in range(100):
        x_adv, num_calls = global_search(src_img, x_adv, soft_edge_mask_adv, model, tar_lbl)
        total_calls += num_calls
        if i < iterations_to_show:
            img_np = denormalize_image(x_adv, mean, std)
            recorded.append((img_np, norm.item()))
        x_adv_inv = inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()), mean, std)
        norm = torch.norm(x_inv - x_adv_inv)
        if norm == previous_norm:
            continuous += 1
            if continuous > 1:
                num_calls = 0
                adv = x_adv
                cln = src_img
                while True:
                    mid = (cln + adv) / 2.0
                    num_calls += 1
                    total_calls += 1
                    if is_adversarial(mid, model, tar_lbl) == 1:
                        adv = mid
                    else:
                        cln = mid
                    if torch.norm(adv - cln).cpu().numpy() < 0.0001 or num_calls >= 25:
                        break
                x_adv = adv
                break
        else:
            continuous = 0
        previous_norm = norm
    if recorded:
        fig, axes = plt.subplots(1, len(recorded), figsize=(4 * len(recorded), 4))
        for idx, (img, dist) in enumerate(recorded):
            ax = axes[idx] if len(recorded) > 1 else axes
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Iteration {idx + 1}")
            ax.text(
                0.5, -0.025,
                f"ℓ² to source  = {dist:.4f}",
                ha='center', va='top',
                transform=ax.transAxes,
                fontsize=10
            )
        plt.tight_layout()
        plt.show()
    print("\n")
    show_image_with_label(x_adv, src_img, mean, std, total_calls, src_lbl=src_lbl_name, tar_lbl=tar_lbl_name, is_adv=1, caption=norm)
    return_arguments = [src_img, x_adv, soft_edge_mask_adv, total_calls, tar_lbl, src_lbl_name, tar_lbl_name, model]
    return return_arguments

def patch_based_edge_informed_search(src_img, x_adv, soft_edge_mask_adv, total_calls, tar_lbl, src_lbl_name, tar_lbl_name, model, model_arch="ViT", file_a=None, file_b=None, iterations_to_show=3):
    model_folder = model_arch.capitalize() if model_arch.lower().startswith('resnet') else model_arch
    soft_edge_mask_adv = soft_edge_mask_adv.unsqueeze(0).unsqueeze(0)
    os.makedirs(model_folder, exist_ok=True)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x_inv = inv_tf(copy.deepcopy(src_img.cpu()[0, :, :, :].squeeze()), mean, std)
    x_adv, patch_calls, patch_boundary_mask = patch_search(src_img, x_adv, soft_edge_mask_adv, model, tar_lbl, mean, std, iterations_to_show=iterations_to_show)
    total_calls += patch_calls
    x_adv_inv = inv_tf(x_adv.cpu().detach()[0, :, :, :].clone(), mean, std)
    norm = torch.norm(x_inv - x_adv_inv)
    print(f'Norm after patch refinement: {norm}, queries used: {total_calls}')
    show_image_with_label(x_adv, src_img, mean, std, total_calls, src_lbl=src_lbl_name, tar_lbl=tar_lbl_name, is_adv=1, caption=norm)
    save_dir = os.path.join('Tensors', model_folder)
    os.makedirs(save_dir, exist_ok=True)
    src_tensor_path = os.path.join(save_dir, file_a.replace('.JPEG', '.pt'))
    adv_tensor_path = os.path.join(save_dir, file_b.replace('.JPEG', '.pt'))
    torch.save(src_img.cpu(), src_tensor_path)
    torch.save(x_adv.cpu(), adv_tensor_path)
    return x_adv, total_calls

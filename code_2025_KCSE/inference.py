import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import zipfile
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# UNet 정의를 포함한 기존 코드에서 UNet 클래스와 기타 관련 코드를 재사용

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(1024 + 512, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 128, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        e5 = self.enc5(nn.MaxPool2d(2)(e4))

        d1 = self.dec1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.sigmoid(self.final(d4))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, image_paths, gt_paths, transform=None, limit = None):
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.transform = transform
        self.limit = limit

        if limit:
            self.image_paths = self.image_paths[:limit]
            self.gt_paths = self.gt_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        input_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
        return (
            torch.tensor(input_image).permute(2, 0, 1).float() / 255.0,
            torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
        )
import matplotlib.pyplot as plt
import torch 
import numpy as np 


# 모델 초기화 및 체크포인트 로드
generator = UNet().to(device)
checkpoint_path = "checkpoint_prompt_1.pth"
load_model = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(load_model["generator_state_dict"])
generator.eval()

# 데이터 로드 함수
def load_images(input_dir):
    image_paths = input_dir#[os.path.join(input_dir, img) for img in sorted(os.listdir(input_dir))]
    images = []

    img = cv2.imread(image_paths)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB로 변환
        img = cv2.resize(img, (256, 256))  # 모델 입력 크기로 리사이즈
        images.append((image_paths, img))
    return images

# 추론 및 저장 함수
def inference_and_save(generator, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = load_images(input_dir)

    for path, img in tqdm(images, desc="Inferencing"):
        input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            output = generator(input_tensor).squeeze(0).cpu().numpy()
        
        # 결과 이미지 저장 (0~255 범위로 변환)
        output_img = (output.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        save_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

# 입력과 출력 경로 설정
input_dir = "/home/work/.hiinnnii/p1_TRAIN_25915.png"
output_dir = "/home/work/.hiinnnii/p1_color.png"

# 추론 실행
inference_and_save(generator, input_dir, output_dir)

print(f"Inference completed. Results saved in {output_dir}")

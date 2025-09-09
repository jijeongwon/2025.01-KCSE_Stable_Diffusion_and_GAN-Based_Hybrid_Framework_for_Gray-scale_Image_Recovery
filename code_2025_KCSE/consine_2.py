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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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
def visualize_predictions(val_loader, generator, device, epoch, num_samples=4):
    generator.eval()
    
    # 배치에서 num_samples만큼 이미지를 선택
    with torch.no_grad():
        for i, (input_images, gt_images) in enumerate(val_loader):
            if i >= 1:  # 첫 번째 배치에서만 출력
                break
            input_images, gt_images = input_images.to(device), gt_images.to(device)
            
            # 모델을 사용해 예측 생성
            fake_images = generator(input_images)

            # 이미지를 numpy 형식으로 변환 (배치에서 첫 번째 이미지)
            input_images = input_images.cpu().numpy()
            gt_images = gt_images.cpu().numpy()
            fake_images = fake_images.cpu().numpy()

            # 시각화
            fig, axs = plt.subplots(num_samples, 3, figsize=(18, num_samples * 6))  # 더 큰 이미지 크기 설정
            for i in range(num_samples):
                # 원본 이미지
                axs[i, 0].imshow(input_images[i].transpose(1, 2, 0), cmap='gray')
                axs[i, 0].set_title('Input Image', fontsize=28)  # 제목 크기 키움
                axs[i, 0].axis('off')

                # Ground Truth 이미지
                axs[i, 1].imshow(gt_images[i].transpose(1, 2, 0), cmap='gray')
                axs[i, 1].set_title('Ground Truth', fontsize=28)  # 제목 크기 키움
                axs[i, 1].axis('off')

                # 예측된 이미지
                axs[i, 2].imshow(fake_images[i].transpose(1, 2, 0), cmap='gray')
                axs[i, 2].set_title('Prediction', fontsize=28)  # 제목 크기 키움
                axs[i, 2].axis('off')

            # 제목 설정
            plt.suptitle(f'Epoch {epoch} Predictions', fontsize=32, fontweight='bold')  # 전체 제목 크기 키움
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # 여백 조정
            plt.subplots_adjust(wspace=0.10, hspace=0.20)  # 이미지 간 간격 조정

            # 그래프를 고해상도로 저장
            plt.savefig(f'output_prompt_2/validation_epoch_{epoch}.png', dpi=300)  # 고해상도 저장
            plt.show()

def save_loss_graph(train_losses, val_losses, epoch, filename='loss_graph.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss up to Epoch {epoch}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def visualize_batch(images, gt_images, num_images=4):
    images = images[:num_images]
    gt_images = gt_images[:num_images]
    
    fig, axs = plt.subplots(2, num_images, figsize=(10, 5))
    for i in range(num_images):
        
        axs[0, i].imshow(images[i], cmap='gray')
        axs[0, i].set_title(f'Image {i+1}')
        axs[0, i].axis('off')
        
        axs[1, i].imshow(gt_images[i], cmap='gray')
        axs[1, i].set_title(f'GT Image {i+1}')
        axs[1, i].axis('off')
    
    plt.show()


input_dir = "/home/work/.Gmama/train_output_2"
gt_dir = "/home/work/.sina/Ptrain_gt"

val_input_dir = "/home/work/.Gmama/val_output_2"
val_gt_dir = "/home/work/.sina/Pval_gt"

train_inputs = [os.path.join(input_dir, img) for img in sorted(os.listdir(input_dir))]
train_gts = [os.path.join(gt_dir, img) for img in sorted(os.listdir(gt_dir))]

val_inputs = [os.path.join(val_input_dir, img) for img in sorted(os.listdir(val_input_dir))]
val_gts = [os.path.join(val_gt_dir, img) for img in sorted(os.listdir(val_gt_dir))]

# # Split into train and validation sets
# train_inputs, val_inputs, train_gts, val_gts = train_test_split(
#     input_images, gt_images, test_size=0.2, random_state=42, shuffle=True
# )

generator = UNet().to(device)
discriminator = PatchGANDiscriminator().to(device)

# generator = nn.DataParallel(generator, device_ids=[0, 1, 2, 3])
# discriminator = nn.DataParallel(discriminator, device_ids=[0, 1, 2, 3])

adversarial_loss = nn.BCELoss()  
pixel_loss = nn.MSELoss()  

optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Create Datasets and DataLoaders
train_dataset = ImageDataset(train_inputs, train_gts)
val_dataset = ImageDataset(val_inputs, val_gts)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

epochs = 30
result_dir = "prompt_2"
os.makedirs(result_dir, exist_ok=True)
checkpoint_path = "checkpoint_prompt_2.pth"


# Define cosine annealing schedulers
# T_max = 10  # One full cycle over all epochs
# scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=T_max, eta_min=1e-4)
#scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=T_max, eta_min=1e-4)

# Initialize lists to store losses
train_losses_G = []
val_losses_G = []


for epoch in range(epochs):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for input_images, gt_images in train_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            real_labels = torch.ones_like(discriminator(gt_images)).to(device)
            fake_labels = torch.zeros_like(discriminator(input_images)).to(device)

            # Train generator
            optimizer_G.zero_grad()
            fake_images = generator(input_images)
            pred_fake = discriminator(fake_images)

            # g_loss_adv = adversarial_loss(pred_fake, real_labels)
            g_loss_pixel = pixel_loss(fake_images, gt_images)
            g_loss = g_loss_pixel
            g_loss.backward()
            optimizer_G.step()

            # Train discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(gt_images)
            loss_real = adversarial_loss(pred_real, real_labels)

            pred_fake = discriminator(fake_images.detach())
            loss_fake = adversarial_loss(pred_fake, fake_labels)

            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()
            pbar.set_postfix(generator_loss=g_loss.item(), discriminator_loss=d_loss.item())
            pbar.update(1)

    # Step schedulers
    # scheduler_gen.step()
    # scheduler_disc.step()

    # Calculate average training loss
    avg_train_loss_G = running_loss_G / len(train_loader)
    train_losses_G.append(avg_train_loss_G)

    # Validation loop
    generator.eval()
    val_loss_G = 0.0
    with torch.no_grad():
        for input_images, gt_images in val_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            fake_images = generator(input_images)
            # g_loss_adv = adversarial_loss(discriminator(fake_images), torch.ones_like(discriminator(fake_images)).to(device))
            g_loss_pixel = pixel_loss(fake_images, gt_images)
            g_loss = g_loss_pixel
            val_loss_G += g_loss.item()

    avg_val_loss_G = val_loss_G / len(val_loader)
    val_losses_G.append(avg_val_loss_G)

    # Print losses and learning rates
    print(f"Epoch [{epoch+1}/{epochs}] - Train Generator Loss: {avg_train_loss_G:.4f}, "
          f"Train Discriminator Loss: {running_loss_D / len(train_loader):.4f}, "
          f"Val Generator Loss: {avg_val_loss_G:.4f}")
    # Save loss graph
    save_loss_graph(train_losses_G, val_losses_G, epoch+1, filename=f'output_prompt_2/loss_epoch_{epoch+1}.png')

    # Visualize predictions for validation set
    visualize_predictions(val_loader, generator, device, epoch+1, num_samples=4)



    print(f"Epoch {epoch+1} valid save")
    test_input_dir = "/home/work/.Gmama/test_output_2"
    output_dir = f"prompt_2_output_images_epoch_{epoch+1}"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for img_name in sorted(os.listdir(test_input_dir)):
            img_path = os.path.join(test_input_dir, img_name)
            img = cv2.imread(img_path)
            input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            output = generator(input_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
            output = output.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, img_name), output)

    zip_filename = os.path.join(result_dir, f"epoch_{epoch+1}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_name in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, img_name), arcname=img_name)
    print(f"Epoch {epoch+1} results saved to {zip_filename}")

    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_path)

#generator.train()  
#discriminator.train()
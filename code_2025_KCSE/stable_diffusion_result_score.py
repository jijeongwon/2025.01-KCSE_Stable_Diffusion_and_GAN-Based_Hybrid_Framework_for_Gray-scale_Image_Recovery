import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def calculate_ssim_mse(original_folder, restored_folder):
    """
    원본 폴더와 복원된 폴더의 이미지 쌍 간 SSIM과 MSE를 계산.
    
    Parameters:
        original_folder (str): 원본 이미지 폴더 경로
        restored_folder (str): 복원된 이미지 폴더 경로

    Returns:
        dict: 각 이미지의 SSIM과 MSE, 전체 평균 SSIM 및 MSE
    """
    original_images = sorted(os.listdir(original_folder))
    restored_images = sorted(os.listdir(restored_folder))

    if len(original_images) != len(restored_images):
        raise ValueError("폴더 내 이미지 수가 일치하지 않습니다.")

    ssim_scores = []
    mse_scores = []

    results = {}
    for original_image, restored_image in tqdm(zip(original_images, restored_images), total=len(original_images)):
        original_path = os.path.join(original_folder, original_image)
        restored_path = os.path.join(restored_folder, restored_image)

        # 이미지 읽기
        img1 = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(restored_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"이미지를 불러올 수 없습니다: {original_image} 또는 {restored_image}")
            continue

        if img1.shape != img2.shape:
            raise ValueError(f"이미지 크기가 일치하지 않습니다: {original_image} vs {restored_image}")

        # SSIM 계산
        ssim_score = ssim(img1, img2)
        ssim_scores.append(ssim_score)

        # MSE 계산
        mse_score = np.mean((img1 - img2) ** 2)
        mse_scores.append(mse_score)

        # 개별 결과 저장
        results[original_image] = {"ssim": ssim_score, "mse": mse_score}

    # 평균 SSIM 및 MSE 계산
    average_ssim = np.mean(ssim_scores)
    average_mse = np.mean(mse_scores)

    results["average_ssim"] = average_ssim
    results["average_mse"] = average_mse

    return results


if __name__ == "__main__":
    # 폴더 경로 설정
    original_folder = "/home/work/.sina/Ptrain_gt_gray"
    restored_folder = "/home/work/jun/train_output_2"

    # SSIM 및 MSE 계산
    results = calculate_ssim_mse(original_folder, restored_folder)

    # 결과 출력
    print(f"전체 평균 SSIM: {results['average_ssim']}")
    print(f"전체 평균 MSE: {results['average_mse']}")

    # # 개별 결과 저장
    # for image_name, metrics in results.items():
    #     if isinstance(metrics, dict):
    #         print(f"{image_name} - SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.4f}")


#train_gt_graty <-> train_output_1 SSIM : 0.6440882618133498 / MSE : 61.018029935963256
#train_gt_graty <-> train_output_2 SSIM : 0.6427843795951725 / MSE : 61.088910419946274
#train_gt_graty <-> train_output_2 SSIM : 0.64291437805744 / MSE : 60.68998535231401
#val_gt_graty <-> val_output_1 SSIM : 0.6475942841282355 / MSE : 61.568054065308225
#val_gt_graty <-> val_output_2 SSIM : 0.6466507987166188 / MSE : 61.574486356516836
#val_gt_graty <-> val_output_2 SSIM : 0.6466783640541429 / MSE : 61.30468642801468
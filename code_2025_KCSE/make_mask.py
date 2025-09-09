import os
import cv2
import numpy as np

# 경로 설정
input_folder = "./Pval_input"
gt_folder = "./Pval_gt_gray"
output_folder = "./Pval_mask"

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 입력 폴더의 파일 처리
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    gt_path = os.path.join(gt_folder, filename)
    
    # 입력 이미지와 원본 이미지 읽기
    input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    if input_image is None or gt_image is None:
        print(f"Failed to read {filename}. Skipping.")
        continue
    
    # 마스크 영역 추출
    mask = cv2.absdiff(input_image, gt_image)
    _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # 윤곽선 감지 및 컨벡스 헐 적용
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask_binary)  # 빈 이미지 생성
    
    for contour in contours:
        hull = cv2.convexHull(contour)  # 컨벡스 헐 계산
        cv2.drawContours(filled_mask, [hull], -1, 255, thickness=cv2.FILLED)  # 컨벡스 헐 내부 채우기

    # 마스크 저장
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, filled_mask)

print(f"프레임 내부가 다양한 모양으로 채워진 마스크 이미지가 {output_folder}에 저장되었습니다.")



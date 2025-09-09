import os
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 컬러 이미지가 저장된 폴더 경로
input_folder = '/home/work/.tmdgy/Ptrain_gt'
# 변환된 흑백 이미지를 저장할 폴더 경로
output_folder = '/home/work/.tmdgy/Ptrain_gt_gray'

# 저장할 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 모든 파일 처리
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 이미지 파일 확장자 확인
        # 이미지 경로
        input_path = os.path.join(input_folder, filename)
        
        # 컬러 이미지 로드
        color_image = cv2.imread(input_path)
        
        # 흑백(그레이스케일) 변환
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # 변환된 이미지 저장 경로
        output_path = os.path.join(output_folder, filename)
        
        # 흑백 이미지 저장
        cv2.imwrite(output_path, gray_image)

print("모든 이미지를 흑백으로 변환 완료!")

import os
import zipfile
from tqdm import tqdm

def compress_folder_to_zip(folder_path, zip_name=None):
    """
    지정된 폴더를 zip 파일로 압축합니다.
    
    Args:
        folder_path (str): 압축할 폴더의 경로
        zip_name (str, optional): 생성할 zip 파일의 이름. 기본값은 폴더 이름에 .zip 확장자를 붙입니다.
    """
    # 폴더가 존재하는지 확인
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
    
    # zip 파일 이름이 지정되지 않은 경우, 폴더 이름을 사용
    if zip_name is None:
        zip_name = folder_path.rstrip('/') + '.zip'
    
    # 폴더 내 모든 파일 수 계산
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    # zip 파일 생성
    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # tqdm으로 진행 상황 표시
            with tqdm(total=len(all_files), desc="압축 진행") as pbar:
                for file_path in all_files:
                    # zip 파일 안에 상대 경로로 추가
                    arcname = os.path.relpath(file_path, start=folder_path)
                    zipf.write(file_path, arcname)
                    pbar.update(1)  # 진행 상황 업데이트
        print(f"압축 완료: {zip_name}")
    except Exception as e:
        print(f"압축 중 오류 발생: {str(e)}")

# 사용 예시
folder_path = '/home/work/.tmdgy/Pval_input'  # 압축할 폴더 경로
compress_folder_to_zip(folder_path)

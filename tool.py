import os
import shutil
from pathlib import Path


def copy_matching_files():
    """
    레이블 디렉터리의 파일 이름 목록을 기준으로,
    소스 이미지 디렉터리에서 일치하는 파일을 찾아
    목적지 이미지 디렉터리로 복사합니다.
    """
    # 1. 경로 설정 (Path 객체 사용)
    # 기준이 되는 파일 이름들이 있는 경로
    labels_dir = Path(r"E:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\A_set\test\labels")

    # 복사할 원본 이미지들이 있는 경로
    source_images_dir = Path(r"E:\LAB\datasets\project_use\CamVid_12_DLC_v1\x4_BILINEAR\images")

    # 파일을 붙여넣을 목적지 경로
    destination_dir = Path(r"E:\LAB\datasets\project_use\CamVid_12_2Fold_LR_x4_Bilinear\A_set\test\images")

    # 2. 목적지 디렉터리 생성 (없을 경우)
    # exist_ok=True: 폴더가 이미 있어도 에러를 발생시키지 않음
    # parents=True: 중간 경로 폴더가 없어도 함께 생성함
    destination_dir.mkdir(parents=True, exist_ok=True)

    print(f"'{labels_dir}' 경로의 파일 목록을 기준으로 복사를 시작합니다.")

    # 3. 파일 복사 작업
    copied_count = 0
    not_found_count = 0

    # labels_dir에 있는 모든 파일 이름을 가져옴
    label_filenames = [f.name for f in labels_dir.iterdir() if f.is_file()]

    if not label_filenames:
        print("기준이 되는 레이블 디렉터리에 파일이 없습니다.")
        return

    for filename in label_filenames:
        # 원본 이미지 파일의 전체 경로 생성
        source_file_path = source_images_dir / filename

        # 대상 이미지 파일의 전체 경로 생성
        destination_file_path = destination_dir / filename

        # 원본 폴더에 파일이 존재하는지 확인
        if source_file_path.exists():
            # 파일 복사 (shutil.copy2는 메타데이터도 함께 복사)
            shutil.copy2(source_file_path, destination_file_path)
            print(f"    복사 완료: {filename}")
            copied_count += 1
        else:
            # 파일이 존재하지 않는 경우
            print(f"    [경고] 원본을 찾을 수 없음: {filename}")
            not_found_count += 1

    # 4. 최종 결과 출력
    print("\n" + "=" * 40)
    print("파일 복사 작업이 완료되었습니다.")
    print(f"    - 성공: {copied_count}개")
    print(f"    - 실패 (파일 없음): {not_found_count}개")
    print(f"    - 복사된 경로: '{destination_dir}'")
    print("=" * 40)


# 스크립트 실행
if __name__ == "__main__":
    copy_matching_files()
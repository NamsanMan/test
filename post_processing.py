import cv2
import matplotlib.pyplot as plt


def canny_edge_detection(image_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Canny 엣지 검출기 적용
    # threshold1, threshold2는 경계를 결정하는 임계값입니다.
    # 이 값들을 조절하여 경계 검출 민감도를 바꿀 수 있습니다.
    canny_edges = cv2.Canny(original_image, threshold1=50, threshold2=150)

    plt.figure(figsize=(12, 6))
    plt.style.use('grayscale')

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(canny_edges)
    plt.title('Canny Edge Map')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- 여기에 분석하고 싶은 이미지 파일 경로를 입력하세요 ---
    image_file = r'E:\LAB\datasets\project_use\CamVid_12_2Fold_v4\A_set\train\images\0001TP_008130.png'
    # ----------------------------------------------------
    canny_edge_detection(image_file)
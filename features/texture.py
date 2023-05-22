from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2

path_1 = "../dataset/train/1_horse/horse_008.jpg"
path_2 = "../dataset/train/2_plane/plane_001.jpg"
path_3 = "../dataset/train/3_rooster/rooster_001.jpg"
path_4 = "../dataset/train/4_tree/tree_001.jpg"
path_5 = "../dataset/train/5_sailboat/sailboat_001.jpg"
path_6 = "../dataset/train/6_motorcycle/motorcycle_001.jpg"
path_7 = "../dataset/train/7_car/car_001.jpg"
path_8 = "../dataset/train/8_butterfly/butterfly_001.jpg"
path_9 = "../dataset/train/9_dragonfly/dragonfly_001.jpg"
path_0 = "../dataset/train/10_flower/flower_001.jpg"

img_path = [path_1, path_2, path_3, path_4, path_5,
            path_6, path_7, path_8, path_9, path_0]

 # Load the image
path = path_1
image = cv2.imread(path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = (gray_image * 255).astype(np.uint8)

# 設定GLCM的參數
distances = [1] # 鄰域距離
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 鄰域方向

# 計算GLCM
glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

# 提取紋理特徵
# 計算對比度-較高的對比度值表示圖像中不同區域之間的亮度差異較大
contrast = graycoprops(glcm, prop='contrast')
# 計算不相似度-較高的不相似度值表示圖像中不同區域之間的灰度差異較大
dissimilarity = graycoprops(glcm, prop='dissimilarity')
#計算均勻度-較高的均勻度值表示圖像中不同區域之間的灰度值較為均勻
homogeneity = graycoprops(glcm, prop='homogeneity')
# #計算能量-較高的能量值表示圖像中灰度值的分佈較平均
# energy = graycoprops(glcm, prop='energy')
#計算相關性-較接近1的相關性值表示圖像中不同區域之間的灰度變化趨勢相似
correlation = graycoprops(glcm, prop='correlation')
# #計算角二階矩值-較高的ASM值表示圖像中灰度值的分佈較平坦
# ASM = graycoprops(glcm, prop='ASM')

texture_array = np.hstack((contrast, dissimilarity, homogeneity, correlation))
texture_array = np.squeeze(texture_array)
texture_list = texture_array.tolist()

print(type(texture_list))
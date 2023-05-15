# # find dominate color
# import matplotlib.image as img
# import matplotlib.pyplot as plt
# from scipy.cluster.vq import whiten
# from scipy.cluster.vq import kmeans
# import pandas as pd
 
# image = img.imread('../dataset/train/5_sailboat/sailboat_093.jpg')
 
# r = []
# g = []
# b = []
# for row in image:
#     for temp_r, temp_g, temp_b in row:
#         r.append(temp_r)
#         g.append(temp_g)
#         b.append(temp_b)
  
# image_df = pd.DataFrame({'red' : r,
#                           'green' : g,
#                           'blue' : b})
 
# image_df['scaled_color_red'] = whiten(image_df['red'])
# image_df['scaled_color_blue'] = whiten(image_df['blue'])
# image_df['scaled_color_green'] = whiten(image_df['green'])
 
# cluster_centers, _ = kmeans(image_df[['scaled_color_red',
#                                     'scaled_color_blue',
#                                     'scaled_color_green']], 
#                             k_or_guess=3, iter=50, seed=5)
 
# dominant_colors = []
 
# red_std, green_std, blue_std = image_df[['red',
#                                           'green',
#                                           'blue']].std()
# print(image_df)
 
# print(cluster_centers)
# for cluster_center in cluster_centers:
#     red_scaled, green_scaled, blue_scaled = cluster_center
#     dominant_colors.append((
#         red_scaled * red_std / 255,
#         green_scaled * green_std / 255,
#         blue_scaled * blue_std / 255
#     ))

 
# plt.imshow([dominant_colors])
# plt.show()

from imagedominantcolor import DominantColor
file_path = '../dataset/train/2_plane/plane_094.jpg'
dominantcolor = DominantColor(file_path)
ans = dominantcolor.dominant_color
print(ans)
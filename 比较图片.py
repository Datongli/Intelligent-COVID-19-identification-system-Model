"""
此文件用于比较p和n两个文件夹中是否有完全相同的图片
"""
import os
from PIL import Image
from PIL import ImageChops
import pandas as pd


csv_path = r"C:\Users\28101\Desktop\重复的.csv"
path_negative = r"C:\Users\28101\Desktop\CCS\negative"
# path_negative = "C:/Users/28101/Desktop/test/n"
path_positive = r"C:\Users\28101\Desktop\CCS\positive"
# path_positive = "C:/Users/28101/Desktop/test/p"
negative_list = os.listdir(path_negative)
positive_list = os.listdir(path_positive)
same_num = 0
for i in range(len(negative_list)):
    image_negative = Image.open(path_negative + "/" + negative_list[i])
    for j in range(len(positive_list)):
        image_positive = Image.open(path_positive + "/" + positive_list[j])
        diff = ImageChops.difference(image_positive, image_negative)
        if diff.getbbox() is None:  # 两张照片如果一样则进行下程序的执行
            same_num += 1
            two_same_image_path = []
            two_same_image_paths = []
            print("-" * 30)
            print(negative_list[i])
            print(positive_list[j])
            print("相同，相同个数为:{}".format(same_num))
            print("-" * 30)
            two_same_image_path.append(negative_list[i])
            two_same_image_path.append(positive_list[j])
            two_same_image_paths.append(two_same_image_path)
            # 一次写入一行
            df = pd.DataFrame(data=two_same_image_paths)
            # 解决追加模式写的表头重复问题
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, header=['negative', 'positive'], index=False, mode='a')
            else:
                df.to_csv(csv_path, header=False, index=False, mode='a')
        else:
            print(negative_list[i])
            print(positive_list[j])
            print("不同")

print("相同的总个数为：{}".format(same_num))




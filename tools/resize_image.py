from PIL import Image
import os

def resize_images(input_folder, output_folder):
    # 创建一个存放修改后图片的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp") or filename.endswith(".gif"):
            img = Image.open(os.path.join(input_folder, filename)) # 打开输入文件夹中的图像文件

            # 以下两个功能选择使用
            # 简单粗暴直接调整图片大小为固定值
            new_size = (1024, 1024)  # 例如：(800, 600) 宽度为800，高度为600

            # 按原图片大小等比例缩放
            # scale = 2.0 # 设置缩放倍数为2倍
            # new_size = tuple(int(dim * scale) for dim in img.size)

            img_resized = img.resize(new_size) # 将图像大小调整为指定的大小
            img_resized.save(os.path.join(output_folder, filename)) # 将调整后的图像保存到输出文件夹中



input_folder = "../datasets/floodnet/val/org_original"
output_folder = "../datasets/floodnet/val/org"

# 执行函数resize_images,调整图片大小并保存到输出文件夹中
resize_images(input_folder, output_folder)
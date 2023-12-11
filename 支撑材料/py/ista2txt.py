# ISAT2YOLO.py

# author：https://github.com/mushroom-x/SegConvert
import json
import os
import shutil

# 定义类别名称与ID号的映射
# 需要注意的是，不需要按照ISAT的classesition.txt里面的定义来
# 可以选择部分自己需要的类别， ID序号也可以重新填写(从0开始)
category_mapping = {"potholes": 0}
# ISAT格式的实例分割标注文件
ISAT_FOLDER = "./annotations"
# YOLO格式的实例分割标注文件
YOLO_FOLDER = "./labels"

# 创建YoloV8标注的文件夹
if not os.path.exists(YOLO_FOLDER):
    os.makedirs(YOLO_FOLDER)


# 载入所有的ISAT的JSON文件
for filename in os.listdir(ISAT_FOLDER):
    if not filename.endswith(".json"):
        # 不是json格式, 跳过
        continue
    # 载入ISAT的JSON文件
    with open(os.path.join(ISAT_FOLDER, filename), "r") as f:
        isat = json.load(f)
    # 提取文件名(不带文件后缀)
    image_name = filename.split(".")[0]
    # Yolo格式的标注文件名, 后缀是txt
    yolo_filename = f"{image_name}.txt"
    # 写入信息
    with open(os.path.join(YOLO_FOLDER, yolo_filename), "w") as f:
        # 获取图像信息
        # - 图像宽度

        image_width = isat["info"]["width"]

        # try:
        #     image_width = isat["info"]["width"]
        # except KeyError:
        #     break
        # - 图像高度
        image_height = isat["info"]["height"]
        # 获取实例标注数据
        for annotation in isat["objects"]:
            # 获取类别名称
            category_name = annotation["category"]
            # 如果不在类别名称字典里面，跳过
            if category_name not in category_mapping:
                continue
            # 从字典里面查询类别ID
            category_id = category_mapping[category_name]
            # 提取分割信息
            segmentation = annotation["segmentation"]
            segmentation_yolo = []
            # 遍历所有的轮廓点
            for segment in segmentation:
                # 提取轮廓点的像素坐标 x, y
                x, y = segment
                # 归一化处理
                x_center = x/image_width
                y_center = y/image_height
                # 添加到segmentation_yolo里面
                segmentation_yolo.append(f"{x_center:.4f} {y_center:.4f}")
            segmentation_yolo_str = " ".join(segmentation_yolo)
            # 添加一行Yolo格式的实例分割数据
            # 格式如下: class_id x1 y1 x2 y2 ... xn yn\n
            f.write(f"{category_id} {segmentation_yolo_str}\n")
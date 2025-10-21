import os
import glob


def rename_files(folder_path):
    # 获取文件夹中所有.png文件
    files = glob.glob(os.path.join(folder_path, "*.png"))

    for file in files:
        # 获取文件名和扩展名
        filename = os.path.basename(file)
        name, ext = os.path.splitext(filename)

        # 检查文件名是否包含“_pixels0”
        if "_pixels0" in name:
            # 构造新的文件名
            new_name = name.replace("_pixels0", "") + ext
            new_file = os.path.join(folder_path, new_name)

            # 重命名文件
            os.rename(file, new_file)
            print(f"Renamed: {filename} -> {new_name}")


if __name__ == "__main__":
    # 指定包含图片的文件夹路径
    folder_path = r"./datasets/NUAA/masks"
    rename_files(folder_path)
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Manager, Queue
import threading

def check_image(file_path):
    """
    检查指定路径的图像文件是否损坏。

    :param file_path: 图像文件的路径
    :return: 如果文件损坏，返回文件路径；否则返回 None
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图像文件是否完整
        return None  # 图像文件没有问题
    except (IOError, SyntaxError):
        return file_path  # 图像文件损坏

def process_folder(folder, progress_queue):
    """
    检查指定文件夹及其子文件夹中的所有 JPEG 文件是否损坏。

    :param folder: 要检查的文件夹路径
    :param progress_queue: 用于更新进度的队列
    :return: 损坏文件的路径列表
    """
    corrupted_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                result = check_image(file_path)
                if result:
                    corrupted_files.append(result)  # 记录损坏的文件路径
                progress_queue.put(1)  # 更新文件进度
    return corrupted_files

def find_corrupted_images(root_dir, num_workers=4):
    """
    查找根目录及其子目录中所有损坏的 JPEG 文件，并返回包含这些文件的文件夹。

    :param root_dir: 根目录路径
    :param num_workers: 并行处理的工作进程数
    :return: 包含损坏文件的文件夹列表
    """
    # 获取所有子文件夹路径
    folders = [os.path.join(root_dir, subdir) for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]
    corrupted_folders = set()  # 用于存储包含损坏文件的文件夹路径

    total_files = sum(len(files) for _, _, files in os.walk(root_dir))

    with Manager() as manager:
        progress_queue = manager.Queue()  # 创建一个队列用于更新进度
        folder_progress = tqdm(total=len(folders), desc="Processing folders")
        file_progress = tqdm(total=total_files, desc="Processing files")

        def update_file_progress():
            while True:
                item = progress_queue.get()
                if item is None:
                    break
                file_progress.update(item)

        progress_thread = threading.Thread(target=update_file_progress)
        progress_thread.start()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有文件夹的处理任务
            futures = {executor.submit(process_folder, folder, progress_queue): folder for folder in folders}
            for future in as_completed(futures):
                folder = futures[future]
                try:
                    corrupted_files = future.result()
                    if corrupted_files:
                        print(f"Corrupted images found in folder: {folder}")
                        corrupted_folders.add(folder)  # 记录包含损坏文件的文件夹路径
                except Exception as exc:
                    print(f"{folder} generated an exception: {exc}")
                folder_progress.update(1)  # 更新文件夹进度条

        progress_queue.put(None)  # 向进度线程发送结束信号
        progress_thread.join()  # 等待进度线程结束

        folder_progress.close()
        file_progress.close()

    return list(corrupted_folders)

if __name__ == "__main__":
    root_dir = "/home/leesin/Develop/DeepLearning/glue-factory/data/revisitop1m/jpg"
    corrupted_folders = find_corrupted_images(root_dir, num_workers=8)  # 使用8个工作进程
    if corrupted_folders:
        print("Folders with corrupted images:")
        for folder in corrupted_folders:
            print(folder)
    else:
        print("No corrupted images found.")

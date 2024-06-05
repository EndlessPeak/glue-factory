# 处理导入相关的工具函数
import importlib.util

from ..utils.tools import get_class
from .base_dataset import BaseDataset

'''
获取数据集
@param name 指定要获取的数据集名称
'''
def get_dataset(name):
    # 可能导入的路径集合，包括传入的 name参数或当前模块的名称
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            # 查找指定路径对应的模块规范
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                # 获取该模块的 BaseDataset 类（基类）
                return get_class(path, BaseDataset)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_dataset__
                except AttributeError as exc:
                    print(exc)
                    continue
    # 指示未找到指定名称的数据集，引发异常
    raise RuntimeError(f'Dataset {name} not found in any of [{" ".join(import_paths)}]')

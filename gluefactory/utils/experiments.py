"""
A set of utilities to manage and load checkpoints of training experiments.

Author: Paul-Edouard Sarlin (skydes)
"""

import logging
import os
import re
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models import get_model
from ..settings import TRAINING_PATH

logger = logging.getLogger(__name__)


def list_checkpoints(dir_):
    """
    List all valid checkpoints in a given directory.
    
    Args:
        dir_ (Path): The directory to search for checkpoints.

    Returns:
        List[Tuple[int, Path]]: A list of tuples where each tuple contains:
            - An integer representing the checkpoint number.
            - A Path object representing the path to the checkpoint file.
    
    Raises:
        AssertionError: 
            If any file in the directory contains more than two numeric groups in its name.
    """
    # 列出给定目录中所有有效的检查点
    checkpoints = []
    # 遍历目录中所有符合 "checkpoint_*.tar" 模式的文件
    for p in dir_.glob("checkpoint_*.tar"):
        # 使用正则表达式查找文件名中的数字
        numbers = re.findall(r"(\d+)", p.name)
        # 使用断言确保找到的数字个数不超过2个
        assert len(numbers) <= 2
        if len(numbers) == 0:
            # 如果文件名中没有找到数字，则跳过该文件
            continue
        if len(numbers) == 1:
            # 如果找到一个数字，将该数字和文件路径添加到检查点列表
            checkpoints.append((int(numbers[0]), p))
        else:
            # 如果找到两个数字，将第二个数字和文件路径添加到检查点列表
            checkpoints.append((int(numbers[1]), p))
    return checkpoints


def get_last_checkpoint(exper, allow_interrupted=True):
    """
    Get the last saved checkpoint for a given experiment name.
    
    Args:
        exper (str): The name of the experiment.
        allow_interrupted (bool, optional): Whether to allow interrupted checkpoints. Defaults to True.

    Returns:
        Path: The path to the last saved checkpoint.
    
    Raises:
        AssertionError: If no valid checkpoints are found.
    """
    # 从指定的实验名称中获取最后保存的检查点
    # list_checkpoints 返回包含所有检查点的列表，每个检查点是一个元组，包含检查点的名称和路径
    ckpts = list_checkpoints(Path(TRAINING_PATH, exper))
    if not allow_interrupted:
        # 过滤掉中断的检查点
        ckpts = [(n, p) for (n, p) in ckpts if "_interrupted" not in p.name]
    # 确保检查点列表非空
    assert len(ckpts) > 0
    # 将检查点列表按名称排序，并返回最后一个检查点的路径
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """
    Get the checkpoint with the best loss, for a given experiment name.

    Args:
        exper (str): The name of the experiment.

    Returns:
        Path: The path to the best saved checkpoint.
    """
    p = Path(TRAINING_PATH, exper, "checkpoint_best.tar")
    return p


def delete_old_checkpoints(dir_, num_keep):
    """Delete all but the num_keep last saved checkpoints."""
    ckpts = list_checkpoints(dir_)
    ckpts = sorted(ckpts)[::-1]
    kept = 0
    for ckpt in ckpts:
        if ("_interrupted" in str(ckpt[1]) and kept > 0) or kept >= num_keep:
            logger.info(f"Deleting checkpoint {ckpt[1].name}")
            ckpt[1].unlink()
        else:
            kept += 1


def load_experiment(exper, conf={}, get_last=False, ckpt=None):
    """
    Load and return the model of a given experiment.
    
    Args:
        exper (str): The path to the experiment directory or checkpoint file.
        conf (dict, optional): Additional configuration to merge. Defaults to {}.
        get_last (bool, optional): Whether to load the last checkpoint if the path is a directory. Defaults to False.
        ckpt (Path, optional): Specific checkpoint file to load. Defaults to None.

    Returns:
        model: The loaded model in evaluation mode.
    """
    # 将 exper 转换为 Path 对象
    exper = Path(exper)
    # 如果 exper 不是 .tar 文件，确定要加载的检查点
    if exper.suffix != ".tar":
        if get_last:
            # 获取最后一个检查点
            ckpt = get_last_checkpoint(exper)
        else:
            # 获取最佳检查点
            ckpt = get_best_checkpoint(exper)
    else:
        ckpt = exper
    logger.info(f"Loading checkpoint {ckpt.name}")
    # 加载检查点
    ckpt = torch.load(str(ckpt), map_location="cpu")

    # 从检查点中加载配置
    loaded_conf = OmegaConf.create(ckpt["conf"])
    OmegaConf.set_struct(loaded_conf, False)
    # 合并配置
    conf = OmegaConf.merge(loaded_conf.model, OmegaConf.create(conf))
    # 获取模型的结构代码模块并设置为评估模式
    # get_model(conf.name) 返回模型类 A
    # A(conf) 会利用完整的配置实例化模型类 
    model = get_model(conf.name)(conf).eval()

    # 加载模型状态字典
    state_dict = ckpt["model"]
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))
    
    # 如果存在缺失的参数，记录警告
    diff = model_params - dict_params
    if len(diff) > 0:
        subs = os.path.commonprefix(list(diff)).rstrip(".")
        logger.warning(f"Missing {len(diff)} parameters in {subs}")
    
    # 将状态字典加载到模型中
    model.load_state_dict(state_dict, strict=False)
    return model


# @TODO: also copy the respective module scripts (i.e. the code)
def save_experiment(
    model,
    optimizer,
    lr_scheduler,
    conf,
    losses,
    results,
    best_eval,
    epoch,
    iter_i,
    output_dir,
    stop=False,
    distributed=False,
    cp_name=None,
):
    """
    Save the current model to a checkpoint and return the best result so far.

    Args:
        model: The model to save.
        optimizer: The optimizer used in training.
        lr_scheduler: The learning rate scheduler used in training.
        conf: The configuration of the experiment.
        losses: The list of losses recorded during training.
        results: The evaluation results of the current epoch.
        best_eval: The best evaluation metric achieved so far.
        epoch: The current epoch number.
        iter_i: The current iteration number within the epoch.
        output_dir: The directory to save the checkpoint.
        stop (bool, optional): Indicates if the training was interrupted. Defaults to False.
        distributed (bool, optional): Indicates if the training is distributed. Defaults to False.
        cp_name (str, optional): The name of the checkpoint file. If None, a name will be generated. Defaults to None.

    Returns:
        float: The best evaluation metric achieved so far.
    """
    # 获取模型的状态字典；如果是分布式训练，则获取子模块的状态字典
    state = (model.module if distributed else model).state_dict()
    # 创建一个字典，包含模型、优化器、学习率调度器的状态，以及配置，当前 Epoch 数量，损失和评估结果
    checkpoint = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "conf": OmegaConf.to_container(conf, resolve=True),
        "epoch": epoch,
        "losses": losses,
        "eval": results,
    }
    # 如果未提供检查点名称，则根据当前 Epoch 和 iteration 生成默认名称
    # 如果训练被中断，则名称包含 _interrupted 字段
    if cp_name is None:
        cp_name = (
            f"checkpoint_{epoch}_{iter_i}" + ("_interrupted" if stop else "") + ".tar"
        )
    logger.info(f"Saving checkpoint {cp_name}")
    # 构建检查点文件路径
    cp_path = str(output_dir / cp_name)
    # 使用 torch.save 将检查点字典保存到文件
    torch.save(checkpoint, cp_path)
    # 如果当前评估结果优于最佳评估结果，且检查点名称并非最佳评估结果，则更新最佳评估结果
    if cp_name != "checkpoint_best.tar" and results[conf.train.best_key] < best_eval:
        best_eval = results[conf.train.best_key]
        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
        shutil.copy(cp_path, str(output_dir / "checkpoint_best.tar"))
    # 删除旧的检查点，只保留最新的指定数量的检查点
    delete_old_checkpoints(output_dir, conf.train.keep_last_checkpoints)
    return best_eval

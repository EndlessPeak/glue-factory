from pathlib import Path

root = Path(__file__).parent.parent  # 项目顶层路径 top-level directory
DATA_PATH = root / "data/"  # 数据集路径 datasets and pretrained weights
TRAINING_PATH = root / "outputs/training/"  # 训练路径 training checkpoints
EVAL_PATH = root / "outputs/results/"  # 评估结果路径 evaluation results

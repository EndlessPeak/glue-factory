import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
import logging

from .models import get_model
from .settings import TRAINING_PATH

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Export matcher model from checkpoint")
    parser.add_argument("experiment", type=str)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.tar",
                        help="Name of the checkpoint file to load (default: checkpoint_best.tar)")
    parser.add_argument("--save-matcher", type=str, default="xfeat_lightglue.pth",
                        help="Name of the file to save the matcher model (e.g., matcher.pth or matcher.pt)")
    args = parser.parse_args()

    # 获取实验目录
    output_dir = Path(TRAINING_PATH, args.experiment)

    # 加载检查点
    checkpoint_path = output_dir / args.checkpoint
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint file {checkpoint_path} does not exist.")
        return

    # 读取检查点中的配置
    init_cp = torch.load(str(checkpoint_path), map_location="cpu")
    conf = OmegaConf.create(init_cp["conf"])
    # del init_cp

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(conf.model.name)(conf.model).to(device)

    # 加载 state_dict 到模型中
    state_dict = init_cp["model"]
    model.load_state_dict(state_dict, strict=False)

    # 打印 matcher 子模型
    if not hasattr(model, "matcher"):
        logging.error("Model does not have a matcher attribute.")
        return
    
    # 获取 matcher 子模型并保存
    matcher = model.matcher
    # print(matcher)
    if args.save_matcher:
        save_path = output_dir / args.save_matcher
        torch.save(matcher.state_dict(), save_path)
        logging.info(f"Matcher model saved to {save_path}")

if __name__ == "__main__":
    main()
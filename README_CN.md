# Glue Factory
Glue Factory 是 CVG 的库，用于训练和评估提取和匹配局部视觉特征的深度神经网络。它能够：

1. 复现点和线匹配的最新模型（如 [LightGlue](https://github.com/cvg/LightGlue)和 [GlueStick](https://github.com/cvg/GlueStick) (ICCV 2023) 的训练。
2. 使用您自己的局部特征或线条在多个数据集上训练这些模型。
3. 在 HPatches 或 MegaDepth-1500 等标准基准上评估特征提取器或匹配器的性能。

# Installation
Glue Factory 运行在 Python 3 和 PyTorch 上。以下是安装该库及其基本依赖项的步骤：

```shell
git clone https://github.com/cvg/glue-factory
cd glue-factory
python3 -m pip install -e .
```

一些高级功能可能需要安装完整的依赖项：
```shell
python3 -m pip install -e .[extra]
```

Glue Factory 中的所有模型和数据集都有自动下载器，因此可以立即开始使用！

# Evaluation
## HPatches
运行评估命令会自动下载 HPatches 数据集，默认下载到 data/ 目录下，需要大约 1.8 GB 的可用磁盘空间。

### LightGlue
评估预训练 SuperPoint 和 LightGlue 模型在 HPatches 上的效果，运行：
```shell
python -m gluefactory.eval.hpatches --conf superpoint+lightglue-official --overwrite
```

我的运行结果如下所示（与官方给出的略有差别）
```text
Tested ransac setup with following results:
AUC {0.5: [0.3507, 0.5734, 0.6934]}
mAA {0.5: 0.5391666666666667}
best threshold = 0.5
{'H_error_dlt@1px': 0.3528,
 'H_error_dlt@3px': 0.6729,
 'H_error_dlt@5px': 0.7759,
 'H_error_ransac@1px': 0.3507,
 'H_error_ransac@3px': 0.5734,
 'H_error_ransac@5px': 0.6934,
 'H_error_ransac_mAA': 0.5391666666666667,
 'mH_error_dlt': nan,
 'mH_error_ransac': 1.094,
 'mnum_keypoints': 1024.0,
 'mnum_matches': 583.5,
 'mprec@1px': 0.336,
 'mprec@3px': 0.941,
 'mransac_inl': 102.0,
 'mransac_inl%': 0.185}
```

默认的鲁棒性估计器是 `opencv`，但我们强烈推荐改用 `poselib`：
```shell
python -m gluefactory.eval.hpatches --conf superpoint+lightglue-official --overwrite \
    eval.estimator=poselib eval.ransac_th=-1
```

设置 `eval.ransac_th=-1` 可通过在一系列阈值上运行评估来自动调整 RANSAC 内点阈值，并报告最佳值的结果。

### GlueStick
评估预训练 SuperPoint 和 GlueStick 模型在 HPatches 上的效果，运行：
```shell
python -m gluefactory.eval.hpatches --conf gluefactory/configs/superpoint+lsd+gluestick.yaml --overwrite

```

由于 GlueStick 使用点和线来求解同调性，因此在这里使用 Hest 这个不同的鲁棒估计器进行估计。

## MegaDepth-1500
运行评估命令会自动下载 MegaDepth-1500 数据集，默认下载到 data/ 目录下，需要大约 1.5 GB 的可用磁盘空间。

### LightGlue
评估预训练 SuperPoint 和 LightGlue 模型，运行：

```shell
python -m gluefactory.eval.megadepth1500 --conf superpoint+lightglue-official
# or the adaptive variant
python -m gluefactory.eval.megadepth1500 --conf superpoint+lightglue-official \
    model.matcher.{depth_confidence=0.95,width_confidence=0.95}
```

改用鲁棒性估计器 `poselib` 时：
```shell
python -m gluefactory.eval.megadepth1500 --conf superpoint+lightglue-official \
    eval.estimator=poselib eval.ransac_th=2.0
```

### GlueStick
评估预训练 SuperPoint 和 GlueStick 模型，运行：
```shell
python -m gluefactory.eval.megadepth1500 --conf gluefactory/configs/superpoint+lsd+gluestick.yaml
```

# Training
通常使用两阶段培训，且在代码中目前它们已合并。

1. 在应用于互联网图像上的大规模合成单应性数据集上进行预训练，使用Oxford-Paris 检索数据集中的100万图像分散集，这需要大约450GB的磁盘空间。

2. 在基于世界各地知名地标的 PhotoTourism 图片的 MegaDepth 数据集上进行微调。该数据集展示了更复杂和真实的外观和视角变化。这需要大约420GB的磁盘空间。

所有训练命令会自动下载数据集。

## LightGlue
使用 SuperPoint 训练 LightGlue。我们首先在单调数据集上预训练 LightGlue：
```shell
python -m gluefactory.train sp+lg_homography # experiment name
    --conf gluefactory/configs/superpoint+lightglue_homography.yaml
```

1. 可以使用其他实验名称代替，默认情况下，检查点将写入 `outputs/training/`
2. 默认批处理大小为 128 ，对应于论文中报告的结果
3. 需要 2 个 3090 GPU，每个 GPU 具有 24GB 的 VRAM，以及 PyTorch >= 2.0 （FlashAttention）
4. 配置由 OmegaConf 管理，因此可以从命令行覆盖任何条目；如果环境为 PyTorch < 2.0 或更弱的 GPU，则可能需要通过以下方式减小批处理大小：

```shell
python -m gluefactory.train sp+lg_homography 
    --conf gluefactory/configs/superpoint+lightglue_homography.yaml  
    data.batch_size=32  # for 1x 1080 GPU
```

请注意，这可能会影响整体性能。可能需要相应地调整学习率。

然后，我们在 MegaDepth 数据集上微调模型：
```shell
python -m gluefactory.train sp+lg_megadepth 
    --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml
    train.load_experiment=sp+lg_homography
```

此处的默认批处理大小为 32。为了加快 MegaDepth 的训练速度，我们建议在训练前缓存本地特征（需要大约 150 GB 的磁盘空间）：
```shell
# extract features
python -m gluefactory.scripts.export_megadepth --method sp --num_workers 8
# run training with cached features
python -m gluefactory.train sp+lg_megadepth
    --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml
    train.load_experiment=sp+lg_homography
    data.load_features.do=True
```

然后，可以使用其实验名称评估模型：
```shell
python -m gluefactory.eval.megadepth1500 --checkpoint sp+lg_megadepth
```

也可以在每个训练周期后使用选项 `--run_benchmarks` 运行所有基准测试。

# Training Plan
目前计划使用开源的局部特征提取器结合 LightGlue 配置。

已经配置 `xfeat-light_{homography}.yaml` 文件，训练计划：
```shell
python -m gluefactory.train xfeat+lg_homography \
    --conf gluefactory/configs/xfeat+lightglue_homography.yaml \ 
    data.batch_size=32  # for 1x 1080 GPU
```

需要配置 `superpoint-open+lightglue_{homography,megadepth}.yaml` 文件。

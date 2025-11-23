# 动手学深度学习 笔记

本仓库存放个人学习《动手学深度学习》的一些个人笔记

## 文件结构

1. doc 文件存放书籍等文档

2. markdown 下存放个人写的一些笔记

3. d2l 是官方仓库一份 submodule

## 环境配置

这里使用 miniconda 进行环境管理，在安装后执行下列命令

```bash
conda create -n d2l python=3.10 -y
conda activate d2l

conda install numpy -y

# No GPU
pip install torch torchvision torchaudio
pip install d2l
```

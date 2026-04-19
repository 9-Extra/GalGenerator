# 简介
一个用来学习图生图模型的仓库，内有多种生成模型的手动实现

# 环境管理
**使用uv进行python环境管理**

使用uv run 运行脚本
使用uv run -m 运行模块的方式执行项目内部入口
使用uv add/remove 管理依赖

**不要**使用系统python，即不要直接用python命令执行脚本，不要用pip安装依赖

# 项目结构

主包名为galgenerator，对应源码位于 src/galgenerator

galgenerator.common用于放置公用代码，其他包为具体的模型实现
每个模型实现包含：网络定义，train模块用于训练，sample模块用于生成
例如使用 uv run -m galgenerator.vae.train --compile 来训练VAE模型

模型的训练和生成结果存放在 runs 目录下


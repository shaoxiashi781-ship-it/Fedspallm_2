这是基于SparseGPT开源代码修改的Fedspallm代码

目前使用OPT系列模型的方法是可以跑通的，示例命令：
python opt.py --model OPT-125m --dataset c4 --sparsity 0.5 --num_clients 4 --epoch 1

python opt.py --model OPT-125m --dataset c4 --sparsity 0.5 --num_clients 4 --epoch 1 

python opt.py OPT-125m c4 --num_clients 4 --global_comm_rounds 1 --local_rounds 1

主要文件解释：
sparsegpt.py - SparseGPT开源代码的原文件，对单个模型的指定层进行剪枝并生成掩码
*RADME_sparsegpt.md - SparseGPT开源代码的文档 
*llama.py - SparseGPT开源代码的原文件，是使用llama系列模型进行单个模型剪枝的代码，未进行过修改
opt.py - 用OPT系列模型执行Fedspallm算法的代码，包括模拟按照客户端资源能力不同分配指定剪枝层，对所有客户端的指定层进行剪枝，掩码聚合，扩展掩码
fedspallm_agg.py - 实现聚合和扩展掩码的文件
REDME.md - 本项目的文档
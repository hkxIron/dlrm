# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import bisect
import builtins
import shutil
import time

# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# onnx
import onnx

# pytorch
import torch
import torch.nn as nn
from numpy import random as ra
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter


# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

exc = getattr(builtins, "IOError", "FileNotFoundError")


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, layer_size_list, sigmoid_layer_index):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, layer_size_list.size - 1):
            cur_layer_size = layer_size_list[i]
            next_layer_size = layer_size_list[i + 1]

            # construct fully connected operator
            linear_layer = nn.Linear(in_features=int(cur_layer_size),
                           out_features=int(next_layer_size),
                           bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (next_layer_size + cur_layer_size))  # np.sqrt(1 / next_layer_size) # np.sqrt(1 / cur_layer_size)
            W = np.random.normal(mean, std_dev, size=(next_layer_size, cur_layer_size)).astype(np.float32)
            std_dev = np.sqrt(1 / next_layer_size)  # np.sqrt(2 / (next_layer_size + 1))
            bt = np.random.normal(mean, std_dev, size=next_layer_size).astype(np.float32)
            # approach 1
            linear_layer.weight.data = torch.tensor(W, requires_grad=True)  # pytorch中layer的weight可以单独拿出来
            linear_layer.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(linear_layer)

            # construct sigmoid or relu operator
            if i == sigmoid_layer_index:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, dim_size:int, vocab_size_list:np.ndarray): # m:2, ln:[2,3,4]
        # ln:[2,3,4], m=2, 意思是分别创建三个embedding矩阵,它们的dim均为2
        emb_module_list = nn.ModuleList()
        for i in range(0, vocab_size_list.size):
            vocab_size = vocab_size_list[i]  # vocab_size
            # construct embedding operator
            embedding_bag = nn.EmbeddingBag(vocab_size, dim_size, mode="sum", sparse=True) # n:vocab_size, m:emb_dim
            # initialize embeddings
            # nn.init.uniform_(embedding_bag.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / vocab_size),
                high=np.sqrt(1 / vocab_size),
                size=(vocab_size, dim_size)
            ).astype(np.float32)
            # approach 1, embedding初始化
            embedding_bag.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # embedding_bag.weight.data.copy_(torch.tensor(W))
            # approach 3
            # embedding_bag.weight = Parameter(torch.tensor(W),requires_grad=True)
            emb_module_list.append(embedding_bag)

        return emb_module_list

    def __init__(
        self,
        embeding_dim=None, # 2
        category_embedding_vocab_sizes=None, # ndarray([4,3,2])
        dense_bottom_layer_size_list=None, # ndarray([4,3,2])
        top_layer_size_list=None, # [8,4,2,1]
        arch_interaction_op=None, # dot
        arch_interaction_itself=False,
        sigmoid_bottom_index=-1,
        sigmoid_top_index=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (embeding_dim is not None)
            and (category_embedding_vocab_sizes is not None)
            and (dense_bottom_layer_size_list is not None)
            and (top_layer_size_list is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create operators
            self.embedding_layers = self.create_emb(embeding_dim, category_embedding_vocab_sizes) # m_spa:2, ln_emb:[4,3,2],此处有3个embedding
            self.bottom_mlp_layers = self.create_mlp(dense_bottom_layer_size_list, sigmoid_bottom_index)
            self.top_mlp_layers = self.create_mlp(top_layer_size_list, sigmoid_top_index)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, sparse_offset, sparse_index, embedding_layers):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        group_embed_layers = [] # K:为embedding的group下标,即这个embedding来源于何处
        for k, sparse_index_group_batch in enumerate(sparse_index):
            sparse_offset_group_batch = sparse_offset[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            current_embed_layer = embedding_layers[k]
            """
            input=index
            input:tensor([0, 1, 1, 3, 2, 3, 0, 2, 1, 2, 1, 2, 3, 2, 0, 1, 2, 2]),input中的元素为vocab_size中的index,个数不定
            offsets:tensor([ 0,  2,  4,  6,  8, 10, 11, 13, 14, 17]), offsets的元素个数即为此次batch大小,即为10
            
            embed_layer(input, offsets)表示:
            offset中[0,2]表示input中的第[0,2)个元素来源于batch中的第0个样本,
            offset中[2,4]表示input中的第[2,4)个元素来源于batch中的第1个样本,
            offset中[4,6]表示input中的第[4,6)个元素来源于batch中的第2个样本,
            ...
            offset中[17,18]表示input中的第[17,18)个元素来源于batch中的第10个样本
            """
            # 此处会调用EmbeddingBag.forward方法, vector:[batch, embedding_dim=2]
            vector = current_embed_layer(input=sparse_index_group_batch, offsets=sparse_offset_group_batch)

            group_embed_layers.append(vector)

        # print(ly)
        return group_embed_layers
    # dense_x:[batch, dim=2], category_embedding:3 list of [batch,dim=2]
    def interact_features(self, dense_out, group_category_embedding_list):
        if self.arch_interaction_op == "dot": # dense与category特征两两交互
            # concatenate dense and sparse features
            (batch_size, dim) = dense_out.shape
            # 此处将dense_out看成同一组embedding, 与category embedding组成新的特征组,进行两两交互
            # group_cate_dense_embed: [batch, feat_group_num=4, dim]
            group_cate_dense_embed = torch\
                .cat([dense_out] + group_category_embedding_list, dim=1) \
                .view((batch_size, -1, dim)) # group_cate_dense_embed: [batch, feat_group_num=4, dim]
            # perform a dot product, 即两两特征组之间进行点乘dot交互
            # transpose(group_cate_dense_embed, 1, 2),
            # Z_interact: [batch, feat_group_num, feat_group_num]
            Z_interact = torch.bmm(group_cate_dense_embed, torch.transpose(group_cate_dense_embed, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z_interact.shape # ni,nj: feat_group_num
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset) # 这种方法更好理解一些

            # approach 2: custom, 任意两组的特征进行两两交互
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)]) # li:[1,2,2,3,3,3]
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)]) # lj:[0,0,1,0,1,2]
            """
            li,lj组成的index如下:
            [[x,_,_],
             [x,x,_], 
             [x,x,x], 
            ]
            """
            # Z_interact:[batch, feat_group_num, feat_group_num],
            # Z_flat:[batch, comb(feat_group_num=4,2)=6], comb(feat_group_num, 2),为两两组合数,代表第li个特征与第lj个特征之间的组合
            Zflat = Z_interact[:, li, lj]
            # 将最后的交互特征与原始dense特征进行拼接后返回
            # concatenate dense features and interactions
            R = torch.cat([dense_out] + [Zflat], dim=1) # [batch, comb(feat_group_num,2)+dense_dim]
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([dense_out] + group_category_embedding_list, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, sparse_offset, sparse_index):
        if self.ndevices <= 1:
            return self.sequential_forward(dense_x, sparse_offset, sparse_index)
        else:
            return self.parallel_forward(dense_x, sparse_offset, sparse_index)

    def sequential_forward(self, dense_x, sparse_offset, sparse_index):
        # 1. dense feature,即连续特征
        # process dense features (using bottom mlp), resulting in a row vector
        dense_out = self.apply_mlp(dense_x, self.bottom_mlp_layers) # dense_out:[batch, embed_dim]
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # 2.离散特征, 生成多个group的batch embedding
        # process sparse features(using embeddings), resulting in a list of row vectors,如: 3个[batch, embed_dim], 3个categroy feature group的个数
        category_embedding_out_list = self.apply_emb(sparse_offset, sparse_index, self.embedding_layers)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # 3. 离散特征之间的交互
        # interact features (dense and sparse), interact:[batch, 8]
        interact = self.interact_features(dense_out, category_embedding_out_list)
        # print(interact.detach().cpu().numpy())

        # obtain probability of a click (using top mlp), 在交互项之后的layer
        p = self.apply_mlp(interact, self.top_mlp_layers) # layer:8->4->2->1

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            prob = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            prob = p

        return prob

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.embedding_layers))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.sync_dense_params or self.parallel_model_is_not_prepared:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bottom_mlp_layers, device_ids)
            self.top_l_replicas = replicate(self.top_mlp_layers, device_ids)
            # distribute embeddings (model parallelism)
            t_list = []
            for k, emb in enumerate(self.embedding_layers):
                d = torch.device("cuda:" + str(k % ndevices))
                emb.to(d)
                t_list.append(emb.to(d))
            self.embedding_layers = nn.ModuleList(t_list)
            self.parallel_model_batch_size = batch_size
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.embedding_layers) != len(lS_o)) or (len(self.embedding_layers) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.embedding_layers):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.embedding_layers)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.embedding_layers) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.embedding_layers):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


if __name__ == "__main__":
    ### import packages ###
    import sys
    import os
    import io
    import collections
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=100)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=10)
    parser.add_argument("--nepochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)

    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    bottom_dense_layer_size_list = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-") # ln_bot:array([4,3,2]), 代表mlp网络:4 ->3 ->2
    print("ln_bot:{}".format(bottom_dense_layer_size_list))
    # input data
    if args.data_generation == "dataset":
        # input and target from dataset
        def collate_wrapper(list_of_tuples):
            # where each tuple is (X_int, X_cat, y)
            transposed_data = list(zip(*list_of_tuples))
            X_int = torch.stack(transposed_data[0], 0)
            X_cat = torch.stack(transposed_data[1], 0)
            T     = torch.stack(transposed_data[2], 0).view(-1,1) # [batch, 1]

            sz0 = X_cat.shape[0]
            sz1 = X_cat.shape[1]
            if use_gpu:
                lS_i = [X_cat[:, i].pin_memory() for i in range(sz1)]
                lS_o = [torch.tensor(range(sz0)).pin_memory() for _ in range(sz1)]
                return X_int.pin_memory(), lS_o, lS_i, T.pin_memory()
            else:
                lS_i = [X_cat[:, i] for i in range(sz1)]
                lS_o = [torch.tensor(range(sz0)) for _ in range(sz1)]
                return X_int, lS_o, lS_i, T

        train_data = dp.CriteoDataset(
            args.data_set,
            args.data_randomize,
            "train",
            args.raw_data_file,
            args.processed_data_file,
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=False,
            drop_last=False,
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(train_loader)

        test_data = dp.CriteoDataset(
            args.data_set,
            args.data_randomize,
            "test",
            args.raw_data_file,
            args.processed_data_file,
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=False,
            drop_last=False,
        )
        nbatches_test = len(test_loader)

        category_embed_vocab_size_list = train_data.counts
        m_den = train_data.m_den
        bottom_dense_layer_size_list[0] = m_den
    else:
        # input and target at random
        def collate_wrapper(list_of_tuples):
            # where each tuple is (X, lS_o, lS_i, T)
            if use_gpu:
                (X, lS_o, lS_i, T) = list_of_tuples[0]
                return (X.pin_memory(),
                        [S_o.pin_memory() for S_o in lS_o],
                        [S_i.pin_memory() for S_i in lS_i],
                        T.pin_memory())
            else:
                return list_of_tuples[0]

        category_embed_vocab_size_list = np.fromstring(args.arch_embedding_size, dtype=int, sep="-") # ln_emb:array([4,3,2])
        m_den = bottom_dense_layer_size_list[0] # bot => bottom, m_den:4
        train_data = dp.RandomDataset(
            m_den, # 4
            category_embed_vocab_size_list, # [4,3,2]
            args.data_size, # 100
            args.num_batches, # 0
            args.mini_batch_size, # 10
            args.num_indices_per_lookup,
            args.num_indices_per_lookup_fixed,
            1, # num_targets
            args.round_targets,
            args.data_generation,
            args.data_trace_file,
            args.data_trace_enable_padding,
            reset_seed_on_access=True,
            rand_seed=args.numpy_rand_seed
        ) #WARNING: generates a batch of lookups at once

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=False,
            drop_last=False,
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(train_loader)

    ### parse command line arguments ###
    embed_dim = args.arch_sparse_feature_size # 2
    # category类特征的个数+1, 但不明白为何要加1
    category_feat_num = category_embed_vocab_size_list.size + 1  # category类特征的个数+1
    #m_den_out = ln_bot[ln_bot.size - 1]
    last_dense_out_dim = bottom_dense_layer_size_list[-1] # dense网络的最终输出维度
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself: # 是否需要与自己交互
            interact_dim = (category_feat_num * (category_feat_num + 1)) // 2 + last_dense_out_dim
        else:
            # 交互项的个数,category为两两交互,而dense与category不用交互, dense与dense也不用交互, dense内部已经交互过
            interact_dim = (category_feat_num * (category_feat_num - 1)) // 2 + last_dense_out_dim
    elif args.arch_interaction_op == "cat": # 不明白concat为啥是num_feat* m_den_out
        interact_dim = category_feat_num * last_dense_out_dim
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    # 将interact放在全连接的第一位
    arch_mlp_top_with_interact = str(interact_dim) + "-" + args.arch_mlp_top  #  8-4-2-1
    top_layer_size_list = np.fromstring(arch_mlp_top_with_interact, dtype=int, sep="-") # [8,4,2,1]
    # sanity check: feature sizes and mlp dimensions must match
    if m_den != bottom_dense_layer_size_list[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(bottom_dense_layer_size_list[0])
        )
    if embed_dim != last_dense_out_dim:
        sys.exit(
            "ERROR: arch-sparse-feature-size "
            + str(embed_dim)
            + " does not match last dim of bottom mlp "
            + str(last_dense_out_dim)
        )
    if interact_dim != top_layer_size_list[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(interact_dim)
            + " does not match first dimension of top mlp "
            + str(top_layer_size_list[0])
        )

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(top_layer_size_list.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(top_layer_size_list)
        print("# of interactions")
        print(interact_dim)
        print(
            "mlp bot arch "
            + str(bottom_dense_layer_size_list.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(bottom_dense_layer_size_list)
        print("# of features (sparse and dense)")
        print(category_feat_num)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(embed_dim)
        print(
            "# of embeddings (= # of sparse features) "
            + str(category_embed_vocab_size_list.size)
            + ", with dimensions "
            + str(embed_dim)
            + "x:"
        )
        print(category_embed_vocab_size_list)

        print("data (inputs and targets):")
        for j, (dense_X, sparse_offset, sparse_index, Y) in enumerate(train_loader):
            # early exit if nbatches was set by the user and has been exceeded
            if j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(dense_X.detach().cpu().numpy())
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(sparse_index[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(sparse_offset)
                ]
            )
            print([S_i.detach().cpu().tolist() for S_i in sparse_index])
            print(Y.detach().cpu().numpy())

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    dlrm = DLRM_Net(
        embed_dim,
        category_embed_vocab_size_list, # 类别类特征的embed list
        bottom_dense_layer_size_list, # 连续类特征的 各层网络节点个数
        top_layer_size_list, # [8,4,2,1]
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bottom_index=-1,
        sigmoid_top_index=top_layer_size_list.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
    )
    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        if ngpus > 1:
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            dlrm.ndevices = min(ngpus, args.mini_batch_size, category_feat_num - 1)
        dlrm = dlrm.to(device)  # .cuda()

    # specify the loss function
    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean") # mean square error
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean") # Binary Cross Entropy
    else:
        sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

    if not args.inference_only:
        # specify the optimizer algorithm
        optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)

    ### main loop ###
    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def dlrm_wrap(dense_x, sparse_offset, sparse_index, use_gpu, device):
        if use_gpu:  # .cuda()
            return dlrm(
                dense_x.to(device),
                [S_o.to(device) for S_o in sparse_offset],
                [S_i.to(device) for S_i in sparse_index],
            )
        else:
            return dlrm(dense_x, sparse_offset, sparse_index)

    def loss_fn_wrap(Y_pred, Y, use_gpu, device):
        if use_gpu:
            return loss_fn(Y_pred, Y.to(device))
        else:
            return loss_fn(Y_pred, Y)

    # training or inference
    best_gA_test = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    k = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved mode {}".format(args.load_model))
        ld_model = torch.load(args.load_model)
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            k = ld_k  # epochs
            j = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0
        print(
            "Saved model Training state: epoch = {:d}/{:d}, batch = {:d}/{:d}, train loss = {:.6f}, train accuracy = {:3.3f} %".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_gL, ld_gA * 100
            )
        )
        print(
            "Saved model Testing state: nbatches = {:d}, test loss = {:.6f}, test accuracy = {:3.3f} %".format(
                ld_nbatches_test, ld_gL_test, ld_gA_test * 100
            )
        )

    print("time/loss/accuracy (if enabled):")
    with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
        while k < args.nepochs:
            # X:[batch, 4], lS_o:3*list of 10 array, lS_i:3 list of [18,16,11], T:10*1
            for j, (dense_X, sparse_offset, sparse_index, Y) in enumerate(train_loader):
                # early exit if nbatches was set by the user and has been exceeded
                if j >= nbatches:
                    break
                '''
                # debug prints
                print("input and targets")
                print(X.detach().cpu().numpy())
                print([np.diff(S_o.detach().cpu().tolist() + list(lS_i[i].shape)).tolist() for i, S_o in enumerate(lS_o)])
                print([S_i.detach().cpu().numpy().tolist() for S_i in lS_i])
                print(Y.detach().cpu().numpy())
                '''
                t1 = time_wrap(use_gpu)

                # TODO:sparse_offset, sparse_index参见函数 apply_emb
                # forward pass, 此处就是net forward
                Y_pred = dlrm_wrap(dense_X, sparse_offset, sparse_index, use_gpu, device)

                # loss forward
                loss = loss_fn_wrap(Y_pred, Y, use_gpu, device)
                '''
                # debug prints
                print("output and loss")
                print(Z.detach().cpu().numpy())
                print(E.detach().cpu().numpy())
                '''
                # compute loss and accuracy
                L = loss.detach().cpu().numpy()  # numpy array, Returns a new Tensor, detached from the current graph.
                Y_pred_vector = Y_pred.detach().cpu().numpy()  # numpy array
                Y = Y.detach().cpu().numpy()  # numpy array
                mini_batch_size = Y.shape[0]  # = args.mini_batch_size except maybe for last
                accurancy = np.sum((np.round(Y_pred_vector, 0) == Y).astype(np.uint8)) / mini_batch_size

                if not args.inference_only:
                    # scaled error gradient propagation
                    # (where we do not accumulate gradients across mini-batches)
                    optimizer.zero_grad()
                    # backward pass
                    loss.backward()
                    # debug prints (check gradient norm)
                    # for l in mlp.layers:
                    #     if hasattr(l, 'weight'):
                    #          print(l.weight.grad.norm().item())

                    # optimizer
                    optimizer.step()

                t2 = time_wrap(use_gpu)
                total_time += t2 - t1
                total_accu += accurancy
                total_loss += L
                total_iter += 1

                print_tl = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                print_ts = (
                    (args.test_freq > 0)
                    and (args.data_generation == "dataset")
                    and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )

                # print time, loss and accuracy
                if print_tl or print_ts:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    gL = total_loss / total_iter
                    total_loss = 0

                    gA = total_accu / total_iter
                    total_accu = 0

                    str_run_type = "inference" if args.inference_only else "training"
                    print(
                        "Finished {} it {}/{} of epoch {}, ".format(
                            str_run_type, j + 1, nbatches, k
                        )
                        + "{:.2f} ms/it, loss {:.6f}, accuracy {:3.3f} %".format(
                            gT, gL, gA * 100
                        )
                    )
                    total_iter = 0

                # testing
                if print_ts and not args.inference_only:
                    test_accu = 0
                    test_loss = 0

                    for jt, (X_test, sparse_offset_test, sparse_index_test, Y_test) in enumerate(test_loader):
                        # early exit if nbatches was set by the user and has been exceeded
                        if jt >= nbatches:
                            break

                        t1_test = time_wrap(use_gpu)

                        # forward pass, 此处用()会调用forward方法
                        Z_test = dlrm_wrap(
                            X_test, sparse_offset_test, sparse_index_test, use_gpu, device
                        )
                        # loss
                        E_test = loss_fn_wrap(Z_test, Y_test, use_gpu, device)

                        # compute loss and accuracy
                        L_test = E_test.detach().cpu().numpy()  # numpy array
                        S_test = Z_test.detach().cpu().numpy()  # numpy array
                        Y_test = Y_test.detach().cpu().numpy()  # numpy array
                        mbs_test = Y_test.shape[
                            0
                        ]  # = args.mini_batch_size except maybe for last
                        A_test = (
                            np.sum((np.round(S_test, 0) == Y_test).astype(np.uint8))
                            / mbs_test
                        )

                        t2_test = time_wrap(use_gpu)

                        test_accu += A_test
                        test_loss += L_test

                    gL_test = test_loss / nbatches_test
                    gA_test = test_accu / nbatches_test

                    is_best = gA_test > best_gA_test
                    if is_best:
                        best_gA_test = gA_test
                        if not (args.save_model == ""):
                            print("Saving model to {}".format(args.save_model))
                            torch.save(
                                {
                                    "epoch": k,
                                    "nepochs": args.nepochs,
                                    "nbatches": nbatches,
                                    "nbatches_test": nbatches_test,
                                    "iter": j + 1,
                                    "state_dict": dlrm.state_dict(),
                                    "train_acc": gA,
                                    "train_loss": gL,
                                    "test_acc": gA_test,
                                    "test_loss": gL_test,
                                    "total_loss": total_loss,
                                    "total_accu": total_accu,
                                    "opt_state_dict": optimizer.state_dict(),
                                },
                                args.save_model,
                            )

                    print(
                        "Testing at - {}/{} of epoch {}, ".format(j + 1, nbatches, 0)
                        + "loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                            gL_test, gA_test * 100, best_gA_test * 100
                        )
                    )

            k += 1  # nepochs

    # profiling
    if args.enable_profiling:
        with open("dlrm_s_pytorch.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("./dlrm_s_pytorch.json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        with open("dlrm_s_pytorch.onnx", "w+b") as dlrm_pytorch_onnx_file:
            (dense_X, sparse_offset, sparse_index, _) = train_data[0] # get first batch of elements
            torch.onnx._export(
                dlrm, (dense_X, sparse_offset, sparse_index), dlrm_pytorch_onnx_file, verbose=True
            )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)

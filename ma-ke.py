import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import argparse
import os
import pdb
from datetime import datetime
from tqdm import tqdm, trange
from scipy.spatial import distance
import scipy.spatial
import shutil
from collections import OrderedDict
import json
import sys

from utils import generate_data

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.is_bias = bias
        self.ent_weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.ent_bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_weight)

    def forward(self, inputs):
        adj, inputs_features = inputs
        output = torch.spmm(adj, inputs_features).mm(self.ent_weight)
        if self.is_bias:
            output += self.ent_bias
        return output

class Model(nn.Module):
    def __init__(self, num_hops, gcn_dims):
        super(Model,self).__init__()
        self.num_hops = num_hops
        self.gcn_dims = gcn_dims

        self.gcn_ent = nn.Sequential(
            GraphConvolution(1024, gcn_dims),
            nn.LeakyReLU()
        )        

        self.gcn_rel = nn.Sequential(
            GraphConvolution(1024, gcn_dims),
            nn.LeakyReLU()
        )

        self.gcn_attr = nn.Sequential(
            GraphConvolution(1024, gcn_dims),
            nn.LeakyReLU()
        )        

        self.gcnx = nn.Sequential(
            GraphConvolution(gcn_dims, gcn_dims),
            nn.LeakyReLU()
        )            

        self.cnn_hops_fusion = nn.Sequential(
            nn.Conv1d(gcn_dims,gcn_dims,num_hops),
            nn.Flatten(),
            nn.LeakyReLU()
        )         

        self.literal_transform = nn.Linear(1024, gcn_dims)
        
        self.gate = nn.Sequential(
            nn.Linear(gcn_dims, gcn_dims),
            nn.Sigmoid()
        )


    def n_hop_repr(self, adj, literal_features):
        for i in range(self.num_hops):
            if i == 0:
                i_hop_repr = self.gcn_ent((adj, literal_features))
            else:
                next_aggregate = self.gcnx((adj, i_hop_repr))
                gate_weight = self.gate(i_hop_repr)
                i_hop_repr = gate_weight * next_aggregate + (1 - gate_weight) * i_hop_repr

        return i_hop_repr    

    def transform_literals(self, literal_features):
        return self.literal_transform(literal_features)   

    def forward(self, ent_features, ent_ids, rel_features, attr_features):
        ent_adj, ent_literal_features = ent_features
        rel_adj, rel_literal_features = rel_features
        attr_adj, attr_literal_features = attr_features
        
        rel_repr = self.gcn_rel((rel_adj, rel_literal_features))
        rel_repr = rel_repr[ent_ids]

        attr_repr = self.gcn_attr((attr_adj, attr_literal_features))
        attr_repr = attr_repr[ent_ids]        
        
        ent_literals = ent_literal_features[ent_ids]
        ent_literals = self.transform_literals(ent_literals)

        n_hop_repr = self.n_hop_repr(ent_adj, ent_literal_features)
        ent_n_hop_repr = n_hop_repr[ent_ids]

        ent_repr = torch.cat([ent_literals, ent_n_hop_repr, rel_repr, attr_repr], dim=1) 
        return ent_repr

class Kernel(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, inputs):
        return self.discriminator(inputs)

def alignment_list(source, target, threshold=None):
    samples_em = source
    target_em = target

    samples_em_normalized = samples_em / samples_em.norm(p=2, dim=1).unsqueeze(-1)
    target_em_normalized = target_em / target_em.norm(p=2, dim=1).unsqueeze(-1)

    scores = samples_em_normalized.mm(target_em_normalized.T)
    top1 = scores.topk(1, dim=1)

    if threshold:
        mask = (top1.values >= threshold).squeeze()
        aligned_list = torch.stack([torch.arange(source.shape[0], device=scores.device)[mask] ,top1.indices.squeeze()[mask]]).T
    else:
        aligned_list = torch.stack([torch.arange(source.shape[0], device=scores.device) ,top1.indices.squeeze()]).T

    return aligned_list         

def MMD(em1, em2, device):
    return (em1.T.mm(torch.ones([em1.size(0), 1]).to(device))/em1.size(0)
                - em2.T.mm(torch.ones([em2.size(0), 1]).to(device))/em2.size(0)).norm(2).pow(2)

def cal_performance_ILL(test_samples, source_nodes, target_nodes, topk=1):
    samples_em = source_nodes[test_samples[:,0]]
    target_em = target_nodes[test_samples[:,1]]

    samples_em_normalized = samples_em / samples_em.norm(p=2, dim=1).unsqueeze(-1)
    target_em_normalized = target_em / target_em.norm(p=2, dim=1).unsqueeze(-1)

    scores = samples_em_normalized.mm(target_em_normalized.T)
    scores = scores - torch.diag(scores).view(-1,1)
    ranks = (scores > 0).sum(dim=1) + 1

    ranks = ranks.cpu().numpy()
    topk_hit = sum(ranks <= topk) * 1.0 / len(ranks)
    mrr = sum([1/r for r in ranks]) / len(ranks)

    return topk_hit, mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dims', type=int, default=300)
    parser.add_argument('--n_negatives', type=int, default=5) 
    parser.add_argument('--subset', default='zh_en')
    parser.add_argument('--cuda', default='1')
    args = parser.parse_args()   

    args.flow_steps = 2000
    args.dropout_prob = 0.0
    args.device = "cuda:{}".format(args.cuda)
    device = args.device

    args.loggingStep = 2
    args.lr = 0.0002
    args.num_hops = 1 
    args.loss_margin = 0.1 

    args.DBP15K_clean_dir = 'datasets/DBP15K_clean'
    args.DBP15K_dir = 'datasets/DBP15k'

    source, target = args.subset.split('_')
    data = generate_data(args)

    train, test = data['train_test']
    train_samples = torch.tensor(train).to(device)
    test_samples = torch.tensor(test).to(device)    

    adj_matrices, embeddings = data[source]
    source_ent_adj = torch.sparse.FloatTensor(torch.tensor(adj_matrices['ent']["i"]),
                                            torch.tensor(adj_matrices['ent']["v"]),
                                            torch.Size(adj_matrices['ent']["shape"])).to(device)  

    source_rel_adj = torch.sparse.FloatTensor(torch.tensor(adj_matrices['rel']["i"]),
                                            torch.tensor(adj_matrices['rel']["v"]),
                                            torch.Size(adj_matrices['rel']["shape"])).to(device) 

    source_attr_adj = torch.sparse.FloatTensor(torch.tensor(adj_matrices['attr']["i"]),
                                            torch.tensor(adj_matrices['attr']["v"]),
                                            torch.Size(adj_matrices['attr']["shape"])).to(device)

    source_ent_literals = torch.tensor(embeddings["ents"]).to(device) 
    source_rel_literals = torch.tensor(embeddings["rels"]).to(device)   
    source_attr_literals = torch.tensor(embeddings["attrs"]).to(device)  

    adj_matrices, embeddings = data[target]
    target_ent_adj = torch.sparse.FloatTensor(torch.tensor(adj_matrices['ent']["i"]),
                                            torch.tensor(adj_matrices['ent']["v"]),
                                            torch.Size(adj_matrices['ent']["shape"])).to(device)  

    target_rel_adj = torch.sparse.FloatTensor(torch.tensor(adj_matrices['rel']["i"]),
                                            torch.tensor(adj_matrices['rel']["v"]),
                                            torch.Size(adj_matrices['rel']["shape"])).to(device) 

    target_attr_adj = torch.sparse.FloatTensor(torch.tensor(adj_matrices['attr']["i"]),
                                            torch.tensor(adj_matrices['attr']["v"]),
                                            torch.Size(adj_matrices['attr']["shape"])).to(device)

    target_ent_literals = torch.tensor(embeddings["ents"]).to(device) 
    target_rel_literals = torch.tensor(embeddings["rels"]).to(device)   
    target_attr_literals = torch.tensor(embeddings["attrs"]).to(device) 


    model = Model(num_hops=args.num_hops, gcn_dims=args.dims)
    model.to(device)

    kernel = Kernel(in_dim=args.dims*4, out_dim=args.dims)
    kernel.to(device)

    hit1, mrr = cal_performance_ILL(test_samples, model.transform_literals(source_ent_literals), model.transform_literals(target_ent_literals))
    print("DBP15K subset: {}".format(args.subset))
    print("Entity name: hit1: %.4f mrr: %.4f" %  (hit1, mrr) )   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    kernel_optimizer = torch.optim.Adam(kernel.parameters(), lr=args.lr)

    time_record = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    log_dir = "make_runs/{}".format(time_record)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)    

    tr_loss, logging_loss = 0.0, 0.0
    flow_iterator = trange(args.flow_steps, desc="Flow Step")

    for flow_step in flow_iterator:
        kernel.zero_grad()

        all_source_repr = model((source_ent_adj, source_ent_literals), torch.arange(source_ent_adj.shape[0]).to(args.device), (source_rel_adj, source_rel_literals), (source_attr_adj, source_attr_literals))
        all_target_repr = model((target_ent_adj, target_ent_literals), torch.arange(target_ent_adj.shape[0]).to(args.device), (target_rel_adj, target_rel_literals), (target_attr_adj, target_attr_literals))           

        all_source_repr_kernel_projected = kernel(all_source_repr.detach())
        all_target_repr_kernel_projected = kernel(all_target_repr.detach())        

        flipped_mmd = -MMD(all_source_repr_kernel_projected, all_target_repr_kernel_projected, args.device)
        flipped_mmd.backward()
        kernel_optimizer.step()

        model.zero_grad()
        all_source_repr_kernel_projected = kernel(all_source_repr)
        all_target_repr_kernel_projected = kernel(all_target_repr)

        mmd = MMD(all_source_repr_kernel_projected, all_target_repr_kernel_projected, args.device)

        source_repr = all_source_repr[train_samples[:,0]]
        target_repr = all_target_repr[train_samples[:,1]]

        source_negative_samples = torch.tensor(np.random.randint(target_ent_literals.shape[0], size=train_samples.shape[0]*args.n_negatives )).to(args.device)
        target_negative_samples = torch.tensor(np.random.randint(source_ent_literals.shape[0], size=train_samples.shape[0]*args.n_negatives )).to(args.device)    

        source_neg_repr = all_target_repr[source_negative_samples]
        target_neg_repr = all_source_repr[target_negative_samples]

        criterion = nn.TripletMarginLoss(margin=args.loss_margin)
        supervised_loss = criterion(source_repr.repeat(args.n_negatives ,1), target_repr.repeat(args.n_negatives ,1), source_neg_repr) + criterion(target_repr.repeat(args.n_negatives ,1), source_repr.repeat(args.n_negatives ,1), target_neg_repr)

        loss = mmd + supervised_loss

        loss.backward()
        optimizer.step()

        tr_loss += loss.item()

        logging_loss = tr_loss
        
        if (flow_step + 1) % 1 == 0:
            top_hit_test, _ = cal_performance_ILL(test_samples, all_source_repr, all_target_repr)
            flow_iterator.set_postfix(loss=loss.item(), hit1=top_hit_test)       

    saved_dir = "saved_data/ma-ke_embeddings/{}".format(args.subset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    torch.save(all_source_repr.detach().cpu(), os.path.join(saved_dir, "{}_embedding_{}.pth".format(source, args.dims*4)))    
    torch.save(all_target_repr.detach().cpu(), os.path.join(saved_dir, "{}_embedding_{}.pth".format(target, args.dims*4)))    

    args.result = { 
        'hit1':  cal_performance_ILL(test_samples, all_source_repr, all_target_repr, topk=1)[0], 
        'hit10': cal_performance_ILL(test_samples, all_source_repr, all_target_repr, topk=10)[0],
        'mrr':  cal_performance_ILL(test_samples, all_target_repr, all_source_repr)[1], 
        }

    with open(os.path.join(log_dir, 'args.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)      
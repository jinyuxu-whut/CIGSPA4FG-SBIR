import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cal_triplet_loss(dis, margin):
    b=dis.shape[0]
    # [b,b]

    eye = torch.eye(b).cuda()
    pos_dis = torch.masked_select(dis, eye.bool())
    # [b]
    loss = pos_dis.unsqueeze(1) - dis + margin
    loss = loss * (1 - eye)
    loss = torch.nn.functional.relu(loss)

    hard_triplets = loss > 0
    # [b,b]
    num_pos_triplets = torch.sum(hard_triplets, dim=1)
    # [b]
    loss = torch.sum(loss, dim=1) / (num_pos_triplets + 1e-16)
    loss = torch.mean(loss)
    return loss

def contextual_loss(sk_emb, ph_emb, match, delta=1.0, topk=10, sigma=3.0):

    bs, n, f = sk_emb.shape
    bp, n, f = ph_emb.shape
    S_dist = match(sk_emb, ph_emb)
    S_dist = S_dist / S_dist.mean(1, keepdim=True)
    
    with torch.no_grad():
        P_dist = match(ph_emb, ph_emb)
        W_P = torch.exp(-P_dist.pow(2) / sigma)
        
        W_P_copy = W_P.clone()

        for i in range(bp):
            W_P_copy[i][i] = 1.

        topk_index = torch.topk(W_P_copy, topk)[1]
        topk_half_index = topk_index[:, :int(np.around(topk/2))]

        W_NN = torch.zeros_like(W_P).scatter_(1, topk_index, torch.ones_like(W_P))

        V = ((W_NN + W_NN.t())/2 == 1).float()

        W_C_tilda = torch.zeros_like(W_P)
        for i in range(bp):
            indNonzero = torch.where(V[i, :]!=0)[0]
            W_C_tilda[i, indNonzero] = (V[:,indNonzero].sum(1) / len(indNonzero))[indNonzero]
            
        W_C_hat = W_C_tilda[topk_half_index].mean(1)
        W_C = (W_C_hat + W_C_hat.t())/2 # symmetric
        W = (W_P + W_C)/2

        identity_matrix = torch.eye(bp).cuda(non_blocking=True)
        pos_weight = identity_matrix   
        neg_weight = W * (1 - identity_matrix)   
        
    pull_losses = torch.relu(S_dist).pow(2) * pos_weight
    push_losses = torch.relu(delta - S_dist).pow(2) * neg_weight
    
    loss = (pull_losses.sum() + push_losses.sum()) / (bs * bs)
    
    return loss
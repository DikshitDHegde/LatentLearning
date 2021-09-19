import os
import pickle
import random

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.utils.data import Dataset
from tqdm import tqdm

def set_seed_globally(seed_value=0,if_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED']=str(seed_value)
    if if_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = True



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def Cluster(args,model,dataset,device,epoch):
    model.eval()
    X,y = dataset

    with torch.no_grad():
    
        X = torch.Tensor(X).to(device).reshape(X.shape[0],-1)
        
        z1,_,_,_ = model(X)

        kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20,n_jobs=-1)
        y_pred1 = kmeans1.fit_predict(z1.data.cpu().numpy())

        # print(y_pred1.shape,y.shape)

        nmi_1 = nmi_score(y_pred1,y)
        acc_1 = cluster_acc(y_pred1,y)
        ari_1 = ari_score(y_pred1,y)

        print(f"Epoch:[{epoch}/{args.epochs}] Latent space  : ACC:{acc_1:.4f} NMI:{nmi_1:.4f} ARI:{ari_1:.4f}")
        k = os.path.join(args.out_dir, "FMNIST", "Latent")
        if not os.path.exists(k):
            os.makedirs(k)
        save_latent(k, epoch+1, z1.detach().cpu().numpy(), y,"Z1")

        return (acc_1,nmi_1,ari_1)


def save_latent(dir, epoch, latent, y,name):
    data = {"latent": latent, "target": y}
    with open(os.path.join(dir, f"{name}_{epoch}_.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



def pretrain(args,model,pretrain_loader,device,optimizer,criterion_mse,epoch):
    model.train()
    loop = tqdm(enumerate(pretrain_loader),total=len(pretrain_loader),leave=False)
    total_recon_loss = 0
    total_emb_loss = 0
    for idx ,(x,_) in loop:
        x = x.reshape(x.shape[0],-1)
        optimizer.zero_grad()
        z, recon, prob1, prob2 = model(x)

        rloss = criterion_mse(recon,x)
        rloss.backward(retain_graph=True)

        pseudo_labels = torch.softmax(prob1/args.thres , dim=-1)
        _,pseudo_labels = torch.max(pseudo_labels,dim=-1)
        EntropyLoss = F.cross_entropy(prob2,pseudo_labels)
        EntropyLoss.backward()
        total_emb_loss ++EntropyLoss.item()*x.shape[0]
        total_recon_loss += rloss.item()*x.shape[0]
        optimizer.step()

        if idx%10==0:
            loop.set_description(f"[{epoch}/{args.epochs}]:")

    return total_emb_loss/70000, total_recon_loss/70000






# pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
# max_probs, targets_u = torch.max(pseudo_label, dim=-1)
# mask = max_probs.ge(args.threshold).float()

# Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

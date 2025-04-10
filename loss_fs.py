import torch
import torch.nn as nn
import torch.nn.functional as F


class SilhouetteDistanceFS(nn.Module):
    """
    PyTorch module to compute Silhouette Distance for few-shot classification
    """

    def __init__(self, return_loss=True, delta=1e-4):
        """
        Args:
            return_loss (bool): If True, returns (1 - score)/2 as loss. Otherwise, returns raw Silhouette score.
            delta (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.return_loss = return_loss
        self.delta = delta

    def forward(self, support_embeddings, support_labels, query_embeddings, query_labels):
        """
        Args:
            support_embeddings (Tensor): Shape (N_support, D)
            support_labels (Tensor): Shape (N_support,)
            query_embeddings (Tensor): Shape (N_query, D)
            query_labels (Tensor): Shape (N_query,)
        Returns:
            Scalar silhouette score or loss
        """
        
        
        
        
        
        
        unique_labels = torch.unique(support_labels)

                


        A, B = self._compute_distances(support_embeddings, support_labels,query_embeddings,query_labels, unique_labels)
        
 
        sil_samples = (B - A) / torch.clamp(torch.maximum(A, B), min=self.delta)

        # nan values are for clusters of size 1, and should be 0
        mean_sil_score = torch.mean(torch.nan_to_num(sil_samples))
        if self.return_loss:
            return (1 - mean_sil_score) / 2
        else:
            return mean_sil_score.item()
        
        
        
        
        
        

    def _compute_distances(self, X_sup, y_sup, X_q, y_q, unique_labels):
        intra_dist = torch.zeros_like(y_q, dtype=torch.float32)
        inter_dist = torch.full_like(y_q, torch.inf, dtype=torch.float32)
    
        for i, label_a in enumerate(unique_labels):
            cluster_indices_a_sup = (y_sup == label_a)
            subX_a_sup = X_sup[cluster_indices_a_sup]
    
            cluster_indices_a_q = (y_q == label_a)
            subX_a_q = X_q[cluster_indices_a_q]
    
            intra_distances_a = torch.cdist(subX_a_sup, subX_a_q)
            intra_dist[cluster_indices_a_q] = intra_distances_a.mean(dim=0)
    
            for label_b in unique_labels[i + 1:]:
                cluster_indices_b_q = (y_q == label_b)
                subX_b_q = X_q[cluster_indices_b_q]
                inter_distances_ab = torch.cdist(subX_a_sup, subX_b_q)
                inter_distances_ba = torch.cdist(subX_b_q, subX_a_sup)
    
                inter_dist[cluster_indices_a_sup] = torch.minimum(inter_distances_ab.mean(dim=1), inter_dist[cluster_indices_a_sup])
                inter_dist[cluster_indices_b_q] = torch.minimum(inter_distances_ba.mean(dim=1), inter_dist[cluster_indices_b_q])
    
        return intra_dist, inter_dist
    
    
    
    
    
    
   


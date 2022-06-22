
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        # print(feats.shape)
        # print(feats)
        # print(x)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        # print(c)
        return feats.view(feats.shape[0], -1), c
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 4, D = 4, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, use_attention=False, num_cell=2): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        self.output_class = output_class
        self.num_cell = num_cell
        assert num_cell in [1, output_class]
        ### 1D convolutional layer that can handle multiple class (including binary)
        
        self.use_attention = use_attention
        if not use_attention:
            self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        else:
             self.ANG = nn.ModuleList()
             for i in range(self.num_cell):
                self.ANG.append(Attn_Net_Gated(L=output_class, D=128, dropout=True, n_classes=1))
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        if not self.use_attention:
            C = self.fcc(B) # 1 x C x 1
        else:
            attn_score = None
            for i in range(self.num_cell):
                Bx = B.transpose(1, 2) # [1, V, C]
                i_attn_score, _ = self.ANG[i](Bx) # [1, V, C] -> [1, V, 1]
                i_attn_score = i_attn_score.transpose(1, 2) # [1, 1, V]
                if attn_score is None:
                    attn_score = i_attn_score # [1, 1, V]
                else:
                    attn_score = torch.cat([attn_score, i_attn_score], dim=1) # [1, 1+1, V]
            assert attn_score.shape[1] == self.output_class # [1, C, V]
            C = (B * attn_score).sum(dim=2) # [1, C, V] -> [1, C, 1]
        C = C.view(1, -1) # [1, C, 1] -> [1, C]
        return C, A, B 
# class BClassifier(nn.Module):
#     def __init__(self, input_size, output_class, dropout_v=0.0): # K, L, N
#         super(BClassifier, self).__init__()
#         self.q = nn.Linear(input_size, 128)
#         self.v = nn.Sequential(
#             nn.Dropout(dropout_v),
#             nn.Linear(input_size, input_size)
#         )
        
#         ### 1D convolutional layer that can handle multiple class (including binary)
#         self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
#     def forward(self, feats, c): # N x K, N x C
#         device = feats.device
#         V = self.v(feats) # N x V, unsorted
#         Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
#         # handle multiple classes without for loop
#         _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
#         m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
#         q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
#         A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
#         A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
#         B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
#         B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
#         C = self.fcc(B) # 1 x C x 1
#         C = C.view(1, -1)
#         # print(C)
#         return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        # prediction_bag, A, B = self.b_classifier(feats, classes)
        prediction_bag = self.b_classifier(feats)
        # return classes, prediction_bag, A, B
        return classes, prediction_bag

class MAXNet(nn.Module):
    def __init__(self, feats_size, num_classes) -> None:
        super(MAXNet, self).__init__()
        self.fc = nn.Linear(feats_size,num_classes)
    
    def forward(self, feats):
        x, _ = torch.max(feats, dim=0)
        x = self.fc(x)
        return x

class MEANNet(nn.Module):
    def __init__(self, feats_size, num_classes) -> None:
        super(MEANNet, self).__init__()
        self.fc = nn.Linear(feats_size,num_classes)
    
    def forward(self, feats):
        x = torch.mean(feats, dim=0)
        x = self.fc(x)
        return x



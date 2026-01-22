import torch
from nflows.nn.nets.resnet import ResidualNet

class LearnableTypeCond(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_feat_dim = cfg.in_feat_dim
        if cfg.use_predictor:
            self.predictor = ResidualNet(
                in_features=cfg.in_feat_dim,
                out_features=40,          
                hidden_features=512,      
                num_blocks=5,             
                activation=torch.nn.ReLU()  
            )
        #using cross entropy loss
        self.predictor_loss = torch.nn.CrossEntropyLoss()
        self.grasp_type_feat = torch.nn.Embedding(
            num_embeddings=40, embedding_dim=cfg.out_feat_dim
        )
        return

    def forward(self, data,global_feature=None,predicted=False):
        if self.cfg.use_predictor and predicted:
                predicted_grasp_prob = self.predictor(global_feature)
                #select the max value
                predicted_grasp_type = torch.argmax(predicted_grasp_prob, dim=1)
                data["grasp_type_id"] = predicted_grasp_type
        if self.cfg.disabled:
            return self.grasp_type_feat(data["grasp_type_id"] * 0)
        else:
            return self.grasp_type_feat(data["grasp_type_id"])

"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
         - pixels: (B, T, C, H, W)
         - action: (B, T, frameskip * action_dim), action_dim = 2 * n_agents
        writes info["emb"] (B, T, D) and info["act_emb"] (B, T, D)
        """
        raise NotImplementedError

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        raise NotImplementedError

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        writes info["predicted_emb"] (B, S, T+1, D)
        """
        raise NotImplementedError

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings.
        returns: (B, S) last-step cost per action candidate
        """
        raise NotImplementedError

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state.
        Called by the stable-worldmodel solvers (CEM / Adam) during MPC.
        """
        raise NotImplementedError

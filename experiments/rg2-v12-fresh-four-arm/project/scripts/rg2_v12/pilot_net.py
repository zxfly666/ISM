import torch
import torch.nn.functional as F
from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig
from ism_diffusion.model import timestep_embedding
MODEL=dict(d_model=128,n_heads=4,n_blocks=7,mlp_ratio=4.,dropout=0.,vocab_size=4,output_classes=2,rope_base=10000.)

class ScaleDenoiser(CoordinateDenseDenoiser):
    def __init__(self):
        super().__init__(CoordinateDenoiserConfig(**MODEL))
        self.scale_embedding=torch.nn.Embedding(2,128)
        torch.nn.init.zeros_(self.scale_embedding.weight)

    def forward(self,tokens,t,coords,level=0):
        b,h,w=tokens.shape
        valid=torch.ones((b,h*w),device=tokens.device,dtype=torch.bool)
        x=self.token_embedding(tokens).reshape(b,h*w,-1)
        levels=torch.full((b,),int(level),device=tokens.device,dtype=torch.long)
        time=self.time_mlp(timestep_embedding(t,self.config.d_model))+self.scale_embedding(levels)
        x=x+time[:,None,:]
        for block in self.blocks:
            x=block(x,time,coords.reshape(b,h*w,2),valid)
        return self.output(F.silu(self.output_norm(x))).reshape(b,h,w,2).permute(0,3,1,2)

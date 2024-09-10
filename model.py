#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
    Model 구조의 선택권 : 1. Linear, 2. Conv
    
    VAE에서의 ... -> Linearity를 고민할 수 있음. -> Manifold mixup in VAE의 latent vector
        추가 term으로 임의의 2개 label (동일 structure)에 대해 manifold mixup 해버리자. (기대하는 것 : 동일한 structure에 대해서 label의 linear한 변화에 따라 latent vector 또한 linear한 변화 -> linear semantic)
"""
# kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
def kldivergence(mu, log_var, target_mu, target_logvar):
    return torch.mean(-0.5 * torch.sum(1 + log_var - target_logvar - ((mu - target_mu)**2 - log_var.exp())/math.exp(target_logvar), dim=1), dim=0)

def fourier_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    r"""Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.
    Returns:
        embedding (torch.Tensor): [N $\times$ dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Encoder1D(nn.Module):
    def __init__(self, input_dim=11, dim=32, latent_dim=8, layer_num=2, p=0.8):
        super().__init__()
        self.layers = nn.ModuleList([])
        input_layer = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.SiLU()
        )
        self.layers.append(input_layer)
        for ln in range(layer_num-1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Dropout(p)
                )
            )
        self.mu_layer = nn.Linear(dim, latent_dim)
        self.log_var_layer = nn.Linear(dim, latent_dim)
    
    def forward(self, x, c):
        inp = torch.concat([x,c[:, None]], dim=1)
        for l in self.layers:
            inp = l(inp)
        mu = self.mu_layer(inp)
        log_var = self.log_var_layer(inp)
        return mu, log_var

class Decoder1D(nn.Module):
    def __init__(self, latent_dim=8, dim=32, out_dim=10, layer_num=2, p=0.6):
        super().__init__()
        self.layers = nn.ModuleList([])
        in_layer = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.SiLU()
        )
        self.layers.append(in_layer)
        for ln in range(layer_num-2):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Dropout(p)
                )
            )
        self.out_layer = nn.Sequential(
            nn.Linear(dim, out_dim)
        ) # sigmoid를 쓸 것인가 ? 고민..
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.out_layer(x)
        

class VAE1D(nn.Module): # Assume : input의 구조에 따라 conditional factor가 달라질 수 있다. -> VAE의 input은 conditional factor + structure 이다.
    def __init__(self, input_dim=11, out_dim=10, dim=32, latent_dim=8, layer_num=4, p=0.6): # input_dim = in_dim + cond_num
        super().__init__()
        self.encoder = Encoder1D(input_dim=input_dim, dim=dim, latent_dim=latent_dim, layer_num = layer_num//2, p=p)
        self.decoder = Decoder1D(latent_dim=latent_dim, dim=dim, out_dim=out_dim, layer_num=layer_num//2, p=p)
    
    
    def encode(self, x, c):
        mu, log_var = self.encoder(x, c)
        return [mu, log_var]
    
    def decode(self, z):
        x = self.decoder(z)
        return x
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)
        return result, x, c, mu, log_var, z
    
    
    def loss_function(self, *args): # mixup을 고려할 것인가?
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = args[4] # Account for the minibatch samples from the dataset
        target_var = args[5]
        recons_loss = F.mse_loss(recons, input)

        kld_loss = kldivergence(mu, log_var, 0, math.log(target_var))
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    
    def latent_sampling(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        return z

class Predictor(nn.Module):
    def __init__(self, input_dim=10, latent_dim=8, dim=[32, 16], condition=1, p=0.6, residual=False, bn=True, concat=False):
        super().__init__()
        self.residual = residual
        self.condition = condition
        output_dim = input_dim
        # if self.condition > 0:
        #     input_dim += self.condition
        #     output_dim = input_dim - self.condition
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        input_layer = nn.Sequential(
            nn.Linear(input_dim, dim[0]),
            nn.SiLU()
        )
        self.enc.append(input_layer)
        if concat==False:
            dim[-1] = latent_dim
        
        for i in range(1, len(dim)):
            if bn:
                self.enc.append(
                    nn.Sequential(
                        nn.Linear(dim[i-1], dim[i]),
                        nn.BatchNorm1d(dim[i]),
                        nn.SiLU(),
                        nn.Dropout(p)
                    )
                )
            else:
                self.enc.append(
                    nn.Sequential(
                        nn.Linear(dim[i-1], dim[i]),
                        nn.SiLU(),
                        nn.Dropout(p)
                    )
                )
            
        dec_dim = dim    
        if concat:
            dec_dim[-1] = dec_dim[-1] + latent_dim
            
                
        for i in range(len(dim)-1):
            if bn:
                self.dec.append(
                    nn.Sequential(
                        nn.Linear(dim[len(dim)-1-i], dim[len(dim)-2-i]),
                        nn.BatchNorm1d(dim[len(dim)-2-i]),
                        nn.SiLU(),
                        nn.Dropout(p)
                    )
                )
            else:
                self.dec.append(
                    nn.Sequential(
                        nn.Linear(dim[len(dim)-1-i], dim[len(dim)-2-i]),
                        nn.SiLU(),
                        nn.Dropout(p)
                    )
                )
        self.dec.append(
            nn.Sequential(
                nn.Linear(dim[0], output_dim), # + activation을 쓸것인가? sigmoid??? -> 고민.. -> 비슷하다 걍 ㅇㅇ
                nn.Sigmoid()
            )
        )
        self.concat = concat
    
    def forward(self, inp, z):
        if self.condition > 0:
            inp, c = inp
            
        hs = []

        
        for i, layer in enumerate(self.enc):
            if self.residual:
                if (i == 0) or (i == (len(self.enc)-1)):
                    inp = layer(inp)
                    hs.append(inp)
                else:
                    inp = layer(inp) + inp
                    hs.append(inp)
            else:    
                inp = layer(inp)
                hs.append(inp)
        if self.concat:
            inp = torch.cat([inp, z], dim=1)
        else:
            inp += z
        
        for i, layer in enumerate(self.dec):
            if self.residual:
                if (i == 0) or (i == (len(self.dec)-1)):
                    inp = layer(inp)
                    hs.append(inp)
                else:
                    inp = layer(inp) + inp
                    hs.append(inp)
            else: 
                inp = layer(inp)
                hs.append(inp)
            
        return inp, hs


# input_dim=11, out_dim=10, dim=32, latent_dim=8, layer_num=4
# layer_num = [4, 4] : Linear Model layer num, VAE layer num

    
    
class VariationalDeepAdjointPredictor(nn.Module):
    def __init__(self, input_dim=10, latent_dim=8, dim=[[32, 16], 32], layer_num=4, condition=1, p=0.6, residual=False, bn=True, concat=True):
        super().__init__()
        
        self.predictor = Predictor(input_dim, latent_dim, dim[0], condition, p, residual, bn, concat)
        self.vae = VAE1D(input_dim=input_dim+condition, out_dim=input_dim, dim=dim[1], latent_dim=latent_dim, layer_num=layer_num, p=p)
        
    def forward(self, x, c):
        recons, _, _, mu, log_var, z = self.vae(x, c)
        adj, _ = self.predictor([x, c], z)
        
        return adj, recons, mu, log_var, z
    
    def loss_function(self, *args, **kwargs): # mixup을 고려할 것인가?
        pred_adj = args[0]
        adj = args[1]
        vae_recons = args[2]
        vae_input = args[3]
        mu = args[4]
        log_var = args[5]
        kld_weight = args[6]
        vae_weight = args[7]
        target_var = args[8]
        
        vae_loss_dict = self.vae.loss_function(vae_recons, vae_input, mu, log_var, kld_weight, target_var)
        vae_loss = vae_loss_dict['loss']
        vae_recon_loss = vae_loss_dict['Reconstruction_Loss']
        vae_kld = vae_loss_dict['KLD']
        
        recon_loss = F.l1_loss(pred_adj, adj)
        
        loss_info = {'total_loss' : recon_loss + vae_weight * vae_loss, 'recon_loss' : recon_loss.detach(),
                     'vae_reconstruction_loss' : vae_recon_loss, "vae_kld" : vae_kld}
        
        return loss_info
        


# nn.Linear : Uniform He Intialization (default)
# gradient vanishing 확인 필요.
# hidden representation 뽑을 수 있게 output 걸어놓기.
class LinearModel(nn.Module):
    def __init__(self, input_dim=10, dim=32, layer_num=4, condition=1, p=0.6, residual=False, bn=True, neural_rep=False, fourier_embed=False):
        super().__init__()
        self.residual = residual
        self.condition = condition
        output_dim = input_dim
        self.dim = dim
        self.neural_rep = neural_rep
        self.fourier_embed = fourier_embed
        if self.condition > 0 and not fourier_embed:
            input_dim += self.condition
            output_dim = input_dim - self.condition
        if self.neural_rep: # 이거 안쓰는게 더 강력하다.
            self.neural = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)
            # self.neural_processing = nn.Sequential( # 이게 같이 들어가면 더 안좋아진다.
            #     nn.Linear(input_dim, input_dim),
            #     nn.SiLU(),
            #     nn.Linear(input_dim, input_dim),
            #     nn.SiLU()
            # )
        self.layers = nn.ModuleList([])
        input_layer = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU()
        )
        self.layers.append(input_layer)
        for ln in range(layer_num-2):
            if bn:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Dropout(p)
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.ReLU(),
                        nn.Dropout(p)
                    )
                )
        self.layers.append(
            nn.Sequential(
                nn.Linear(dim, output_dim) # + activation을 쓸것인가? sigmoid??? -> 고민.. -> 비슷하다 걍 ㅇㅇ
            )
        )
    
    def forward(self, inp):
        if self.condition > 0:
            inp, c = inp
            if not self.fourier_embed:
                inp = torch.cat([inp, c[:, None]], dim=1)
        if self.neural_rep:
            inp += self.neural_rep
            # inp = self.neural_processing(inp)
        hs = []
        for i, layer in enumerate(self.layers):
            if self.residual:
                if (i == 0) or (i == (len(self.layers)-1)):
                    inp = layer(inp)
                    hs.append(inp)
                else:
                    inp = layer(inp) + inp
                    hs.append(inp)
            else:    
                if i == 0 and self.fourier_embed:
                    inp= layer(inp) + fourier_embedding(c, dim=self.dim)
                else:    
                    inp = layer(inp)
                    hs.append(inp)
            
        return inp, hs


class ResidualBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.flag = False
        if indim != outdim:
            self.flag = True
        
        self.res = nn.Sequential(
            nn.BatchNorm1d(indim),
            nn.SiLU(),
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.SiLU(),
            nn.Linear(outdim, outdim)
        )
        if self.flag:
            self.skip = nn.Linear(indim, outdim)
        
    def forward(self, x):
        out = self.res(x)
        if self.flag:
            skip = self.skip(x)
        else:
            skip = x
        return out + skip
            
            

class SingleResidualLinearModel(nn.Module): # dim을 변경할건지.. 일단 변경 없이로...
    def __init__(self, input_dim=10, dim=32, layer_num=4):
        super().__init__()
        output_dim = input_dim
        
        self.stem = nn.Linear(input_dim, dim)
    
        self.layers = nn.ModuleList([])
        for i in range(layer_num):
            self.layers.append(ResidualBlock(dim, dim))
        
        self.head = nn.Linear(dim, output_dim)
        
    def forward(self, inp):
        x = self.stem(inp)
        for l in self.layers:
            x = l(x)
        return self.head(x)
        
class CondProcessingLinearModel(nn.Module):
    def __init__(self, input_dim=10, dim=128, layer_num=4, condition=1, p=0.6, residual=False, bn=True, neural_rep=True, concat=True):
        super().__init__()
        self.residual = residual
        self.condition = condition
        output_dim = input_dim
        self.neural_rep = neural_rep
        self.bn = bn
        # if concat:
        #     input_dim += self.condition
        #     output_dim = input_dim - self.condition
            
        
        if self.neural_rep:
            self.neural = nn.Parameter(torch.randn(1, input_dim))
            
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.prior = nn.ModuleList([])
        
        prior_dim = dim//2
        self.prior.append(
            nn.Sequential(
                nn.Linear(condition, prior_dim),
                nn.LeakyReLU()
            )
        )
        for i in range(layer_num-2):
            if self.bn:
                self.prior.append(
                    nn.Sequential(
                        nn.Linear(prior_dim, prior_dim//2),
                        nn.BatchNorm1d(prior_dim//2),
                        nn.LeakyReLU(),
                        nn.Dropout(p)
                    )
                )
            else:
                self.prior.append(
                    nn.Sequential(
                        nn.Linear(prior_dim, prior_dim//2),
                        nn.LeakyReLU(),
                        nn.Dropout(p)
                    )
                )
            prior_dim = prior_dim//2
        self.prior.append(
            nn.Linear(prior_dim, condition)
        )
        
        
        input_layer = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LeakyReLU()
        )
        self.enc.append(input_layer)
        for ln in range(layer_num-1):
            if bn:
                self.enc.append(
                    nn.Sequential(
                        nn.Linear(dim, dim//2),
                        nn.BatchNorm1d(dim//2),
                        nn.LeakyReLU(),
                        nn.Dropout(p)
                    )
                )
            else:
                self.enc.append(
                    nn.Sequential(
                        nn.Linear(dim, dim//2),
                        nn.LeakyReLU(),
                        nn.Dropout(p)
                    )
                )
            dim = dim//2
            
        for ln in range(layer_num-1):
            if (ln == 0) and concat:
                if bn:
                    self.dec.append(
                        nn.Sequential(
                            nn.Linear(dim+condition, dim*2),
                            nn.BatchNorm1d(dim*2),
                            nn.LeakyReLU(),
                            nn.Dropout(p)
                        )
                    )
                else:
                    self.dec.append(
                        nn.Sequential(
                            nn.Linear(dim+condition, dim*2),
                            nn.LeakyReLU(),
                            nn.Dropout(p)
                        )
                    )
            else:
                if bn:
                    self.dec.append(
                        nn.Sequential(
                            nn.Linear(dim, dim*2),
                            nn.BatchNorm1d(dim*2),
                            nn.LeakyReLU(),
                            nn.Dropout(p)
                        )
                    )
                else:
                    self.dec.append(
                        nn.Sequential(
                            nn.Linear(dim, dim*2),
                            nn.LeakyReLU(),
                            nn.Dropout(p)
                        )
                    )
            dim = dim*2
        self.dec.append(
            nn.Sequential(
                nn.Linear(dim, output_dim), # + activation을 쓸것인가? sigmoid??? -> 고민.. -> 비슷하다 걍 ㅇㅇ
                nn.Sigmoid()
            )
        )
        self.concat = concat
    
    def forward(self, inp):
        if self.condition > 0:
            inp, c = inp
            c = c[:, None]
            for prior in self.prior:
                c = prior(c)
        if self.neural_rep:
            inp += self.neural_rep
            
        encs = []
        for e in self.enc:
            inp = e(inp)
            encs.append(inp)
            
        if self.concat:
            inp = torch.cat([inp, c], dim=1)
        else:
            inp += c
        

        for i, d in enumerate(self.dec):
            if i > 0 and i < len(self.dec)-1:
                inp = d(inp) + encs[len(encs)-i-2]
            else:
                inp = d(inp)
            
        return inp       

class Unet1d(nn.Module):
    def __init__(self, input_dim=1, dim=16, condition=1):
        super().__init__()
        self.condition = condition
        output_dim = input_dim
        
        if self.condition > 0:
            input_dim += self.condition
            output_dim = input_dim - self.condition
            
        self.down_sample = nn.MaxPool1d(2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
        
        self.input_layer = nn.Sequential(
            nn.Conv1d(input_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.down1 = nn.Sequential(
            nn.Conv1d(dim, dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv1d(dim*2, dim*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim*4, dim*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.down3 = nn.Sequential(
            nn.Conv1d(dim*4, dim*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim*4, dim*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        
        
        self.mid_blocks = nn.Sequential(
            nn.Conv1d(dim*4, dim*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim*4, dim*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        
        self.up1 = nn.Sequential(
            nn.Conv1d(dim*8, dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
        )
        
        self.up2 = nn.Sequential(
            nn.Conv1d(dim*4, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        
        self.up3 = nn.Sequential(
            nn.Conv1d(dim*2, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.output_layer = nn.Conv1d(dim, output_dim, kernel_size=1)
    
    def match_size_concat(self, x1, x2):
        B, C, L1 = x1.shape
        B, C, L2 = x2.shape
        pad_size = L1 - L2
        x2 = F.pad(x2, (0, pad_size), mode='constant', value=0)
        return torch.cat([x1, x2], dim=1)
    
            
    def forward(self, inp):
        if self.condition > 0:
            inp, c = inp # c : BS, 
            if len(inp.shape) == 2:
                inp = inp[:, None, :] # BS, 1, L
            L = inp.shape[-1]
            c = c[:, None, None].repeat(1, 1, L)
            inp = torch.cat([inp, c], dim=1) # BS, 2, L
            
        h1 = self.input_layer(inp) # BS, 1, L -> BS, C, L
        h1_ = self.down_sample(h1) # BS, C, L -> BS, C, L//2
        h2 = self.down1(h1_) # BS, C, L//2 -> BS, C*2, L//2
        h2_ = self.down_sample(h2) # BS, C*2, L//2 -> BS, C*2, L//4
        h3 = self.down2(h2_) # Bs, C*2, L//4 -> BS, C*4, L//4
        h3_ = self.down_sample(h3) # BS, C*4, L//4 -> BS, C*4, L//8
        h4 = self.down3(h3_)# BS, C*4, L//4 -> BS, C*4, L//4
        
        u_h3 = self.up1(self.match_size_concat(h3, self.up_sample(h4)))
        u_h2 = self.up2(torch.cat([h2, self.up_sample(u_h3)], dim=1)) # BS, C*4, L//2 -> BS, C, L //2
        u_h1 = self.up3(torch.cat([h1, self.up_sample(u_h2)], dim=1)) # BS, C*2, L -> BS, C, L
        return self.output_layer(u_h1)[:, 0, :], 0
        
        
        
        

# assume : Locality 존재
# conv1d : (B, 10, 1) -> stride = 1, kernel = 3, pad = 1
class ConvLinearModel(nn.Module):
    def __init__(self, input_dim=10, dim=8, layer_num=4, condition=1, p=0.6, residual=False, bn=True, neural_rep=False):
        super().__init__()
        self.residual = residual
        self.condition = condition
        output_dim = input_dim
        self.neural_rep = neural_rep
        
        if self.condition > 0:
            input_dim += self.condition
            output_dim = input_dim - self.condition
        if self.neural_rep: # 이거 안쓰는게 더 강력하다.
            self.neural = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)
            # self.neural_processing = nn.Sequential( # 이게 같이 들어가면 더 안좋아진다.
            #     nn.Linear(input_dim, input_dim),
            #     nn.SiLU(),
            #     nn.Linear(input_dim, input_dim),
            #     nn.SiLU()
            # )
        
        self.layers = nn.ModuleList([])
        input_layer = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )
        self.layers.append(input_layer)
        for ln in range(layer_num-2):
            if bn:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(dim),
                        nn.SiLU(),
                        nn.Dropout(p)
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
                        nn.SiLU(),
                        nn.Dropout(p)
                    )
                )
        self.last_layer= nn.Sequential(
                nn.Linear(dim*input_dim, output_dim), # + activation을 쓸것인가? sigmoid??? -> 고민.. -> 비슷하다 걍 ㅇㅇ
                nn.Sigmoid()
            )
    
    def forward(self, inp):
        if self.condition > 0:
            inp, c = inp
            inp = torch.cat([inp, c[:, None]], dim=1)
            # inp = self.neural_processing(inp)
        hs = []
        BS, C = inp.shape
        inp = inp.view(BS, -1, C)
        for i, layer in enumerate(self.layers):
            if self.residual:
                if (i == 0) or (i == (len(self.layers)-1)):
                    inp = layer(inp)
                    hs.append(inp)
                else:
                    inp = layer(inp) + inp
                    hs.append(inp)
            else:    
                inp = layer(inp)
                hs.append(inp)
        inp = self.last_layer(inp.view(BS, -1))
            
        return inp, hs
    
    
class FourierBlock(nn.Module):
    def __init__(self, indim, outdim, bias=False):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.bias= bias
        self.scale = (1/ (indim * outdim))
        self.weights = nn.Parameter(self.scale * torch.rand(indim//2+1, outdim//2+1, dtype=torch.cfloat))
        if self.bias:
            self.bias = nn.Parameter(torch.zeros(1, outdim//2+1, dtype=torch.cfloat))
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bi,io->bo", input, weights) 

    def forward(self, x):
        bs = x.shape[0]
        x_ft = torch.fft.rfft(x)
        
        out_ft = self.compl_mul1d(x_ft, self.weights) + self.bias
        x = torch.fft.irfft(out_ft)
        return x

class FNO(nn.Module):
    def __init__(self, indim=10, dim=32, layer_num=4, condition_num=0):
        super().__init__()
        if condition_num > 0:
            bias = True
        else:
            bias = False
            
        self.condition_num = condition_num
        self.indim = indim
        self.dim = dim
        self.layer_num = layer_num
        
        self.stem = nn.Linear(indim+condition_num, dim)
        
        self.fnos = nn.ModuleList([])
        self.projs = nn.ModuleList([])
        
        for i in range(layer_num):
            self.fnos.append(FourierBlock(dim, dim, bias))
            self.projs.append(nn.Linear(dim, dim))
            
        self.head = nn.Linear(dim, indim)
        
    def forward(self, x):
        if self.condition_num > 0:
            x, l = x
            x = torch.concat([x, l[:, None]], dim=1)
        x = self.stem(x)
        
        for fno, proj in zip(self.fnos, self.projs):
            x1 = fno(x)
            x2 = proj(x)
            x = x1 + x2
            x = F.silu(x)
        x = self.head(x)
        
        return x




class ResidualFNO(nn.Module):
    def __init__(self, indim=10, dim=32, layer_num=4, condition_num=0):
        super().__init__()
        
        self.condition_num = condition_num
        self.indim = indim
        self.dim = dim
        self.layer_num = layer_num
        
        self.stem = nn.Linear(indim+condition_num, dim)
        
        self.fnos = nn.ModuleList([])
        self.projs = nn.ModuleList([])
        
        for i in range(layer_num):
            self.fnos.append(FourierBlock(dim, dim))
            self.projs.append(ResidualBlock(dim, dim))
            
        self.head = nn.Linear(dim, indim)
        
    def forward(self, x):
        if self.condition_num > 0:
            x = torch.concat(x, dim=1)
        x = self.stem(x)
        
        for fno, proj in zip(self.fnos, self.projs):
            x1 = fno(x)
            x2 = proj(x)
            x = x1 + x2
            x = F.silu(x)
        x = self.head(x)
        
        return x

    
class NewFourierBlock(nn.Module):
    def __init__(self, indim, outdim, mode, bias=False):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.bias= bias
        self.mode = mode
        self.scale = (1/ (indim * outdim))
        self.weights = nn.Parameter(self.scale * torch.rand(indim, outdim, mode, dtype=torch.cfloat))
        if self.bias:
            self.bias = nn.Parameter(torch.zeros(1, outdim, mode, dtype=torch.cfloat))
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # batch, c, m
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bim,iom->bom", input, weights) 

    def forward(self, x):
        bs = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        x_ft_ = self.compl_mul1d(x_ft[:, :, :self.mode], self.weights) + self.bias
        out_ft[:, :, :self.mode] = x_ft_
        x = torch.fft.irfft(out_ft)
        return x

class NewFNO(nn.Module):
    def __init__(self, indim=10, dim=32, mode=30, layer_num=4, condition_num=0):
        super().__init__()
        if condition_num > 0:
            bias = True
        else:
            bias = False
        
        indim = 1
        self.condition_num = condition_num
        self.indim = indim
        self.dim = dim
        self.layer_num = layer_num
        
        self.stem = nn.Linear(indim, dim)
        
        self.fnos = nn.ModuleList([])
        self.projs = nn.ModuleList([])
        
        for i in range(layer_num):
            self.fnos.append(NewFourierBlock(dim, dim, mode, bias))
            self.projs.append(nn.Linear(dim, dim))
            
        self.head = nn.Linear(dim, indim)
        
    def forward(self, x):
        # x : BS, N
        
        if self.condition_num > 0:
            x, l = x
            cond = fourier_embedding(l, dim=self.dim)
        else:
            cond = 0
        x = x[:, :, None] # x : BS, N, 1
        x = self.stem(x).permute(0, 2, 1) + cond[:, :, None] # BS, C, N + BS, C, 1
        
        for fno, proj in zip(self.fnos, self.projs):
            x1 = fno(x)
            x2 = proj(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x1 + x2
            x = F.silu(x)
        # BS, C, N
        x = self.head(x.permute(0, 2, 1)) # BS, N, C
        
        x = x[:, :, 0] # BS, 1, N -> BS, N
        return x
    

# class UNet1D(nn.Module):
#     def __init__(self, input_dim=10, condition=1, p=0.6, residual=False, bn=True):

# %%

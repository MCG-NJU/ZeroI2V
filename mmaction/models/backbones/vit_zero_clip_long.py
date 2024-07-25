from collections import OrderedDict
from mmengine.model.weight_init import trunc_normal_
import torch
from torch import nn
import clip
from mmengine.logging import MMLogger
from einops import rearrange
from mmaction.registry import MODELS
from ..common import STDHA_long


class LinearAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, scale=1., dropout_rate=0.1):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features) # TODO 分组先不实现
        self.dropout = nn.Dropout(dropout_rate)  #, inplace=True)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale_act = None
        if isinstance(scale, str):
            raise NotImplementedError(scale)
        else:
            self.scale = scale
        

    def forward(self, x):
        # x is (HW+1, BT, D)
        xs = self.D_fc1(x)
        xs = self.dropout(xs)
        xs = self.D_fc2(xs)
        # if self.scale_act is not None:
        #     scale = self.scale_act(self.scale)
        # else:
        #     scale = self.scale
        if self.skip_connect:
            x = x + xs * self.scale
        else:
            x = xs * self.scale
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock_stdha(nn.Module):
    adapter_map = {'rep': LinearAdapter} # , 'gelu': GELUAdapter}
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1,
                 num_frames=8, dropout_rate=0.1, adapter_type='rep', mlp_ratio=0.25, stdha_cfg=dict(), attn_adapter_type='share'):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.stdha_cfg = stdha_cfg
        self.attn = STDHA_long(embed_dim=d_model, num_heads=n_head, 
                num_frames=num_frames,shift_pattern=self.stdha_cfg.get('shift_pattern', 'kv'),
                 divide_head=self.stdha_cfg.get('divide_head', False),
                  shift1_div=self.stdha_cfg.get('shift1_div', 12), 
                  shift2_div=self.stdha_cfg.get('shift2_div', -1), 
                  shift3_div=self.stdha_cfg.get('shift3_div', -1), 
                  shift3_right=self.stdha_cfg.get('shift3_right', False),
                  shift4_div=self.stdha_cfg.get('shift4_div', -1), 
                  shift4_right=self.stdha_cfg.get('shift4_right', False),
                 )
        
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        logger = MMLogger.get_current_instance()
        logger.info(f'num_tadapter:{num_tadapter}, mlp_ratio:{mlp_ratio} scale:{scale} dropout_rate:{dropout_rate} stdha_cfg: {stdha_cfg}')


        self.MLP_Adapter = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate)
        if self.num_tadapter == 2:
            self.MLP_Adapter_out = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate)

        if self.num_tadapter != -1:
            self.S_Adapter = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate)
        
        self.attn_adapter_type = attn_adapter_type
        if attn_adapter_type == 'share':
            self.T_Adapter_in = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate)
        else:
            raise NotImplementedError(self.attn_adapter_type)

        self.num_frames = num_frames
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gradcam_spatial = nn.Identity() 
        self.gradcam_temporal = nn.Identity()


    def attention(self, x: torch.Tensor):
        if self.attn_adapter_type == 'share':
            x = self.T_Adapter_in(x)
            q = k = v = x
        else:
            raise NotImplementedError(self.attn_adapter_type)
        

        x[:, :, :448] = self.gradcam_temporal(x[:, :, :448]) # 7 theads
        x[:, :, 448:] = self.gradcam_spatial(x[:, :, 448:])
        
        return self.attn(q, k, v, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        # spatial adaptation with shift
        if self.num_tadapter != -1:
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
        else:
            x = x + self.attention(self.ln_1(x))
        # joint adaptation
        if self.num_tadapter == 2:
            x = x + self.MLP_Adapter_out(self.mlp(self.MLP_Adapter(self.ln_2(x))))
        else:
            x = x + self.mlp(self.MLP_Adapter(self.ln_2(x)))
        return x

    
class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., dropout_rate=0.1, adapter_type='rep', mlp_ratio=0.25, stdha_cfg=dict(), attn_adapter_type='share'):
        super().__init__()
        self.width = width
        self.layers = layers
        # dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_stdha(width, heads, attn_mask, scale, num_tadapter, num_frames, dropout_rate=dropout_rate, adapter_type=adapter_type, mlp_ratio=mlp_ratio, stdha_cfg=stdha_cfg, attn_adapter_type=attn_adapter_type) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_Zero_CLIP_long(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int,
     dropout_rate=0.1, num_tadapter=1, adapter_scale=0.5, pretrained=None, adapter_type='rep', mlp_ratio=0.25, stdha_cfg=dict(), final_ta_cfg=dict(), attn_adapter_type='share'):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, dropout_rate=dropout_rate, adapter_type=adapter_type, mlp_ratio=mlp_ratio, stdha_cfg=stdha_cfg, attn_adapter_type=attn_adapter_type)

        self.final_ta_cfg = final_ta_cfg
        if self.final_ta_cfg.get('use_it', False):
            raise NotImplementedError
            # if self.final_ta_cfg.get('type') == 'full_tuning':
            #     self.final_ta = GlobalAttentionPool(d_model=width, n_head=heads, num_frames=num_frames)
            # elif self.final_ta_cfg.get('type') == 'adapter_tuning':
            #     self.final_ta = AdapterGlobalAttentionPool(d_model=width, n_head=heads, num_frames=num_frames, adapter_type=self.final_ta_cfg.get('adapter_type', 'rep'))
            # else:
            #     raise NotImplementedError(self.final_ta_cfg.get('type'))
            
        self.ln_post = LayerNorm(width)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = MMLogger.get_current_instance()
            
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                logger.info(f'load model from: {self.pretrained} clip ViT-B/16')
                clip_model, _ = clip.load("ViT-B/16", device="cpu", download_root=self.pretrained)
            else:
                logger.info(f'load model from: {self.pretrained} clip ViT-L/14')
                clip_model, _ = clip.load("ViT-L/14", device="cpu", download_root=self.pretrained)
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            raise NotImplementedError('why do not you use the clip pretrained model?')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


        if self.final_ta_cfg.get('use_it', False): # NOTE 复用最后一层预训练权重
            msg = self.final_ta.load_state_dict(self.transformer.resblocks[-1].state_dict(), strict=False)
            logger.info('Missing keys for final_ta: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys for final_ta: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully for final_ta'{self.pretrained}'")
        ## initialize Adapter
        for n, m in self.transformer.named_modules():
            if 'Adapter' in n or 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


        ## freeze some parameters
        for name, param in self.named_parameters():
            if self.final_ta_cfg.get('use_it', False) and self.final_ta_cfg.get('type') == 'full_tuning' and 'final_ta' in name:
                param.requires_grad = True
            elif 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    

    def forward(self, x: torch.Tensor):
        ## Space-only
        B, C, T, H, W = x.shape
        assert T == self.num_frames, f"{T} != {self.num_frames}"
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 0] # 取cls token ND

        x = self.ln_post(x)

        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x

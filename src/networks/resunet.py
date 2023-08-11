import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.blocks import UpConv, DownConv, ECABlock
from einops import rearrange

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)

class CoarseEncoder(nn.Module):
    def __init__(self, in_channels=3, depth=3, blocks=1, start_filters=32, residual=True, norm=nn.BatchNorm2d, act=F.relu):
        super(CoarseEncoder, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)  
            # pooling = True if i < depth-1 else False
            pooling = True
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def forward(self, x):       
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)      
        
        return x, encoder_outs

class SharedBottleNeck(nn.Module):
    def __init__(self, in_channels=512, depth=5, shared_depth=2, start_filters=32, blocks=1, residual=True,
                 concat=True,  norm=nn.BatchNorm2d, act=F.relu, dilations=[1,2,5]):
        super(SharedBottleNeck, self).__init__()
        self.down_convs = []
        self.up_convs = []

        self.down_im_atts = []
        self.down_mask_atts = []
        self.up_im_atts = []
        self.up_mask_atts = []

        dilations = [1,2,5]
        start_depth = depth - shared_depth
        max_filters = 512
        for i in range(start_depth, depth): 
            ins = in_channels if i == start_depth else outs
            outs = min(ins * 2, max_filters)
            # Encoder convs
            pooling = True if i < depth-1 else False 
            down_conv = DownConv(ins, outs, blocks, pooling=pooling, residual=residual, norm=norm, act=act, dilations=dilations)
            self.down_convs.append(down_conv)

            # Decoder convs
            if i < depth - 1:
                up_conv = UpConv(min(outs*2, max_filters), outs, blocks, residual=residual, concat=concat, norm=norm,act=F.relu,dilations=dilations)
                self.up_convs.append(up_conv)

                self.up_im_atts.append(ECABlock(outs))
                self.up_mask_atts.append(ECABlock(outs))
       
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        # task-specific channel attention blocks
        self.up_im_atts = nn.ModuleList(self.up_im_atts)
        self.up_mask_atts = nn.ModuleList(self.up_mask_atts)


        self.to_qkv_im = nn.Linear(256,256*3)
        self.to_qkv_mask = nn.Linear(256,256*3)

        self.softmax = nn.Softmax(dim=-1)

        reset_params(self)


    def self_attention(self,input,to_qkv):
        n,c,h,w = input.size()
        heads = 4

        input = input.reshape(n,c,h*w).transpose(1,2) 
        qkv = to_qkv(input).chunk(3,dim=-1) 
        q,k,v = map(lambda t: rearrange(t, 'b n (heads d) -> b heads n d',heads = heads),qkv) 

        scale = c**(-0.5)
        dots = torch.matmul(q,k.transpose(-1,-2))*scale 
        attn = self.softmax(dots)  
        out = torch.matmul(attn,v) 
        out = rearrange(out,'b heads n d -> b n (heads d)') 
        out=out.transpose(1,2).reshape(n,c,h,w) 

        return out    

    def forward(self, input):
        # Encoder convs 
        im_encoder_outs = []
        mask_encoder_outs = []
        x = input
        for i, d_conv in enumerate(self.down_convs):  
            # d_conv, attn = nets
            x, before_pool = d_conv(x)
            im_encoder_outs.append(before_pool)
            mask_encoder_outs.append(before_pool)   # im_encoder_outs,mask_encoder_outs 
        x_im = x  
        x_mask = x  

        encoder_feature = im_encoder_outs[0]
        # Decoder convs
        x = x_im
        for i, nets in enumerate(zip(self.up_convs, self.up_im_atts)): 
            up_conv, attn = nets
            before_pool = None
            if im_encoder_outs is not None:
                before_pool = im_encoder_outs[-(i+2)]  
            x = up_conv(x, before_pool,se=attn)
        x_im = x  

        x = x_mask       
        for i, nets in enumerate(zip(self.up_convs, self.up_mask_atts)):
            up_conv, attn = nets
            before_pool = None
            if mask_encoder_outs is not None:
                before_pool = mask_encoder_outs[-(i+2)] 
            x = up_conv(x, before_pool, se = attn)
        x_mask = x  
    
        x_im = self.self_attention(x_im,self.to_qkv_im) + x_im
        x_mask = self.self_attention(x_mask,self.to_qkv_mask) +x_mask
        
        return x_im, x_mask, encoder_feature 

class CoarseDecoder(nn.Module):
    def __init__(self, args, in_channels=512, out_channels=3, norm='bn',act=F.relu, depth=5, blocks=1, residual=True,
                 concat=True, use_att=False):
        super(CoarseDecoder, self).__init__()
        self.up_convs_bg = []
        self.up_convs_mask = []

        # apply channel attention to skip connection for different decoders
        self.atts_bg = []
        self.atts_mask = []
        self.use_att = use_att
        outs = in_channels
        for i in range(depth): 
            ins = outs
            outs = ins // 2
            # background reconstruction branch
            up_conv = UpConv(ins, outs, blocks, residual=residual, concat=concat, norm=norm,act=F.relu)
            self.up_convs_bg.append(up_conv)
            if self.use_att:
                self.atts_bg.append(ECABlock(outs))
            
            # mask prediction branch
            up_conv = UpConv(ins, outs, blocks, residual=residual, concat=concat, norm=norm,act=F.relu)
            self.up_convs_mask.append(up_conv)
            if self.use_att:
                self.atts_mask.append(ECABlock(outs))
        # final conv
        self.conv_final_bg = nn.Conv2d(outs, out_channels, 1,1,0)
        self.conv_ac_final_mask =nn.Sequential(
            *[
                nn.Conv2d(outs, 1, 1,1,0),
                nn.Sigmoid()
            ]
        )
        
        self.up_convs_bg = nn.ModuleList(self.up_convs_bg)
        self.atts_bg = nn.ModuleList(self.atts_bg)
        self.up_convs_mask = nn.ModuleList(self.up_convs_mask)
        self.atts_mask = nn.ModuleList(self.atts_mask)
        
        reset_params(self)

    def forward(self, bg, mask, encoder_outs=None):
        bg_x = bg
        mask_x = mask
        bg_outs = []
        for i, up_convs in enumerate(zip(self.up_convs_bg, self.up_convs_mask)):  
            up_bg, up_mask = up_convs
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+1)] 

            if self.use_att:
                mask_before_pool = self.atts_mask[i](before_pool)
                bg_before_pool = self.atts_bg[i](before_pool)
            
            mask_x = up_mask(mask_x, mask_before_pool)
            bg_x = up_bg(bg_x, bg_before_pool)
            bg_outs.append(bg_x)

        if self.conv_final_bg is not None:
            bg_x = self.conv_final_bg(bg_x)
            mask = self.conv_ac_final_mask(mask_x)
            bg_outs = [bg_x] + bg_outs
        
        return bg_outs, [mask]


#################################################################
#           Refinement Stage
#################################################################

class Refinement(nn.Module):
    def __init__(self, in_channels=4,out_channels=1,ngf = 32, blocks=3, n_skips=3):
        # in_channels =4,out_channels = 3, shared_depth = 1,
        super(Refinement, self).__init__()

        self.n_skips = n_skips

        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, ngf, 3,1,1), nn.InstanceNorm2d(ngf), nn.LeakyReLU(0.2))
        self.down1 = DownConv(ngf, ngf, blocks, pooling=True, residual=True, norm=nn.BatchNorm2d, act=F.relu)
        self.down2 = DownConv(ngf, ngf*2, blocks, pooling=True, residual=True, norm=nn.BatchNorm2d, act=F.relu)
        self.down3 = DownConv(ngf*2, ngf*4, blocks, pooling=True, residual=True, norm=nn.BatchNorm2d, act=F.relu)
        self.down4 = DownConv(ngf*4, ngf*8, blocks, pooling=False, residual=True, norm=nn.BatchNorm2d, act=F.relu)

        self.up1 = UpConv(ngf*8, ngf*4, blocks, residual=True, concat=True, norm='bn', act=F.relu )
        self.up2 = UpConv(ngf*4, ngf*2, blocks, residual=True, concat=True, norm='bn', act=F.relu )
        self.up3 = UpConv(ngf*2, ngf, blocks, residual=True, concat=True, norm='bn', act=F.relu)

        self.dec_conv2 = nn.Sequential(nn.Conv2d(ngf*1,ngf*1,1,1,0))
        self.dec_conv3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*1,1,1,0), nn.LeakyReLU(0.2), nn.Conv2d(ngf, ngf, 3,1,1), nn.LeakyReLU(0.2))
        self.dec_conv4 = nn.Sequential(nn.Conv2d(ngf*4,ngf*2,1,1,0), nn.LeakyReLU(0.2), nn.Conv2d(ngf*2, ngf*2, 3,1,1), nn.LeakyReLU(0.2))

        self.out_conv = nn.Sequential(*[
            nn.Conv2d(ngf, ngf, 3,1,1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, out_channels, 1,1,0)
        ])     
        
    def forward(self, coarse_bg, mask, decoder_outs):
        if self.n_skips < 1:
            dec_feat2 = 0
        else:
            dec_feat2 = self.dec_conv2(decoder_outs[0])

        if self.n_skips < 2:
            dec_feat3 = 0
        else:
            dec_feat3 = self.dec_conv3(decoder_outs[1]) # 64

        if self.n_skips < 3:
            dec_feat4 = 0
        else:
            dec_feat4 = self.dec_conv4(decoder_outs[2]) # 64

        xin = torch.cat([coarse_bg, mask], dim=1)
        x = self.conv_in(xin)

        x,d1 = self.down1(x + dec_feat2) 
        x,d2 = self.down2(x + dec_feat3) 
        x,d3 = self.down3(x + dec_feat4) 
        x,d4 = self.down4(x)    

        x = self.up1(x,d3) 
        x = self.up2(x,d2)  
        x = self.up3(x,d1)  

        im = self.out_conv(x)
        return im   

class SLBR(nn.Module):
    def __init__(self, args, in_channels=3, depth=5, shared_depth=2, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True,
                 concat=True, long_skip=False):
        super(SLBR, self).__init__()
        self.shared = shared_depth = 2
        self.optimizer_encoder,  self.optimizer_image, self.optimizer_wm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        self.args = args
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks) 
        
        # coarse stage
        self.encoder = CoarseEncoder(in_channels=in_channels, depth= depth - shared_depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, norm='bn',act=F.relu)
        self.shared_decoder = SharedBottleNeck(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=depth, shared_depth=shared_depth, blocks=blocks[4], residual=residual,
                                                concat=concat, norm='in')
        
        self.coarse_decoder = CoarseDecoder(args, in_channels=start_filters * 2 ** (depth - shared_depth),
                                        out_channels=out_channels_image, depth=depth - shared_depth,
                                        blocks=blocks[1], residual=residual, 
                                        concat=concat, norm='bn', use_att=True,
                                        )

        self.long_skip = long_skip
        
        # refinement stage
        if args.use_refine:
            self.refinement = Refinement(in_channels=4, out_channels=3, n_skips=args.k_skip_stage)
        else:
            self.refinement = None
        
    def forward(self, synthesized, get_feature = False):
        # Step1: encoder                         
        image_code, before_pool = self.encoder(synthesized)             
        unshared_before_pool = before_pool 

        # Step2: shared_decoder->coarse_decoder
        im, mask,encoder_feature = self.shared_decoder(image_code) 
        if get_feature:
            return encoder_feature

        im_base_feature = im
        mask_base_feature = mask
        ims, mask = self.coarse_decoder(im, mask, unshared_before_pool)
        im = ims[0]  
        reconstructed_image = torch.tanh(im)
        if self.long_skip:
            reconstructed_image = (reconstructed_image + synthesized).clamp(0,1)

        reconstructed_mask = mask[0]            
        
        # Step3: refinement
        if self.refinement is not None:
            dec_feats = (ims)[1:][::-1]            
            coarser = reconstructed_image * reconstructed_mask + (1-reconstructed_mask)* synthesized          
            refine_bg = self.refinement(coarser, reconstructed_mask, dec_feats)            
            refine_bg = (torch.tanh(refine_bg) + synthesized).clamp(0,1)                                    
            return [refine_bg, reconstructed_image], mask,im_base_feature,mask_base_feature
        else:
            return [reconstructed_image], mask, im_base_feature,mask_base_feature  

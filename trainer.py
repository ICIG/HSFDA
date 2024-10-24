import torch.nn as nn
import torch, time, torchaudio
import pandas as pd
import os, sys
from tqdm import tqdm
import numpy as np
from utils.util import get_filepaths, check_folder, cal_score, get_feature, progress_bar
from utils.eval_composite import eval_composite
# from model.res_LSTM import res_LSTM


class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader,Test_path, writer, model_path, score_path, args):
#         self.step = 0
        self.epoch = epoch
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer
        
        # fea = {
        #     'BLSTM'  :'log1p',
        #     'DPTNet' :'wav'
        # }

        fea = {
            'odconv'  :'log1p',
            'DPTNet' :'wav',
            'BLSTM':'log1p',
            'odconv_revise'  :'log1p',
            'odconv2d' : 'log1p',
            'res_LSTM' : 'log1p',
            'CBAM' : 'log1p',
            'sConformer' : 'log1p',
            'new_conformer': 'log1p',
            'new_conformer_1': 'log1p',
            'new_conformer_2': 'log1p',
            'new_conformer_3': 'log1p',
            'new_conformer_4': 'log1p',
            'new_conformer_5': 'log1p',
            'new_conformer_6': 'log1p',
            'new_conformer_7': 'log1p',
            'new_conformer_7re': 'log1p',
            'new_conformer_7res': 'log1p',
            'new_conformer_7res2conv': 'log1p',
            'new_conformer_7res_conv': 'log1p',
            'new_conformer_7res_layer6': 'log1p',
            'new_conformer_7res_layer6_2ssl': 'log1p',
            'new_conformer_7res_layer6_2sslw': 'log1p',
            'new_conformer_7res_feature': 'log1p',
            'new_conformer_7res_fastcnn': 'log1p',
            'new_conformer_7se': 'log1p',
            'new_conformer_8': 'log1p',
            'new_conformer_9': 'log1p',
            'new_conformer_7res2': 'log1p',
            'new_conformer_7res3': 'log1p',
            'new_conformer_7res4': 'log1p',
            'new_conformer_7res5': 'log1p',
            'new_conformer_10': 'log1p',
            'new_conformer_11': 'log1p',
            'new_conformer_12': 'log1p',
            'new_conformer_13': 'log1p',
            'test_1conv': 'log1p',
            'FAodconv': 'log1p',
            'FAodconv01': 'log1p',
            'conformer_res_7_conv': 'log1p',
            
         }        
        self.fea   = fea[args.model]

        self.device = device
        self.loader = loader
        self.criterion = criterion
        self.Test_path = Test_path

        self.train_loss = 0
        self.val_loss = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.args = args
        self.transform = get_feature()
        if not args.finetune_SSL and args.feature!='raw':
            self.model.model_SSL.eval()
            for name,param in self.model.model_SSL.named_parameters():
                param.requires_grad = False         
        if args.finetune_SSL=='PF':
            for name,param in self.model.model_SSL.feature_extractor.named_parameters():
                param.requires_grad = False            

    def save_checkpoint(self,):
        save_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state_dict = {
            'epoch': self.epoch,
            'model': save_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path_)
        torch.save(state_dict, self.model_path_)
    
    def get_fea(self,wav,ftype='log1p'):
        if ftype=='wav':
            return wav
        elif ftype=='complex':
            return self.transform(wav,ftype=ftype)[0]
        else:
            return self.transform(wav,ftype=ftype)[0][0]
        
    def _train_step(self, nwav,cwav):
        device = self.device
        nwav,cwav = nwav.to(device),cwav.to(device)
        # print("cwav",cwav.shape)

        cdata = self.get_fea(cwav, ftype=self.fea)
        # print("cdata",cdata.shape)
        pred = self.model(nwav)
        loss = self.criterion(pred,cdata)
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#             if USE_GRAD_NORM:
#                 nn.utils.clip_grad_norm_(self.model['discriminator'].parameters(), DISCRIMINATOR_GRAD_NORM)
#             self.optimizer['discriminator'].step()

    def _train_epoch(self):
        self.train_loss = 0
        self.model.train()
        if not self.args.finetune_SSL and self.args.feature!='raw':
            self.model.model_SSL.eval()
        step = 0
        t_start =  time.time()
        for nwav,cwav in self.loader['train']:
            #(batchsize,N,T,2)
            self._train_step(nwav,cwav)
            step += 1
            progress_bar(self.epoch,self.epochs,step,self.train_step,time.time()-t_start,loss=self.train_loss,mode='train')
        self.train_loss /= len(self.loader['train'])
        print(f'train_loss:{self.train_loss}')

    
#     @torch.no_grad()
    def _val_step(self, nwav,cwav):
        device = self.device
        nwav,cwav = nwav.to(device),cwav.to(device)
        cdata = self.get_fea(cwav, ftype=self.fea)
        pred = self.model(nwav)
        loss = self.criterion(pred,cdata)
        self.val_loss += loss.item()


    # def _val_epoch(self):
    #     self.val_loss = 0
    #     self.model.eval()
    #     step = 0
    #     t_start =  time.time()
    #     for nwav,cwav in self.loader['val']:
    #         self._val_step(nwav,cwav)
    #         step += 1
    #         progress_bar(self.epoch,self.epochs,step,self.val_step,time.time()-t_start,loss=self.val_loss,mode='test')
    #     self.val_loss /= len(self.loader['val'])
    #     print(f'val_loss:{self.val_loss}')
        
    #     if self.best_loss > self.val_loss:
            
    #         print(f"Save model to '{self.model_path}'")
    #         self.save_checkpoint()
    #         self.best_loss = self.val_loss
    
    def _val_epoch(self):
        self.val_loss = 0
        self.model.eval()
        step = 0
        t_start =  time.time()
        for nwav,cwav in self.loader['val']:
            self._val_step(nwav,cwav)
            step += 1
            progress_bar(self.epoch,self.epochs,step,self.val_step,time.time()-t_start,loss=self.val_loss,mode='test')
        self.val_loss /= len(self.loader['val'])
        print(f'val_loss:{self.val_loss}')
    
        if self.best_loss > self.val_loss and self.epoch <= 15:
            self.model_path_ = self.model_path+'_pre15.pth.tar'
            print(f"Save model to '{self.model_path_ }'")
            self.save_checkpoint()
            self.best_loss = self.val_loss

        if self.best_loss > self.val_loss and self.epoch > 15:
            self.model_path_ = self.model_path+'_'+str(self.epoch)+'.pth.tar'
            print(f"Save model to '{self.model_path_ }'")
            self.save_checkpoint()
            self.best_loss = self.val_loss
    
            
    def write_score(self,test_file,c_path):
        args = self.args
        wavname = test_file.split('/')[-1]
        c_file  = os.path.join(c_path,wavname)
        n_data,sr = torchaudio.load(test_file)
        c_data,sr = torchaudio.load(c_file)

        enhanced  = self.model(n_data.to(self.device),output_wav=True)
        out_path = f'/home/zt/ssl/base_demo_BSSE-SE/exp/Enhanced/{self.model.__class__.__name__}_{args.model}_{args.ssl_model}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}_{args.feature}_{args.size}_'\
                    f'WS{args.weighted_sum}_FT{args.finetune_SSL}/{wavname}'

             
                    
        check_folder(out_path)
        enhanced = enhanced.cpu()
        torchaudio.save(out_path,enhanced,sr)
            
        s_pesq, s_stoi = cal_score(c_data.squeeze().detach().numpy(),enhanced.squeeze().detach().numpy())
        date = eval_composite(c_data.squeeze().detach().numpy(),enhanced.squeeze().detach().numpy())
        with open(self.score_path_, 'a') as f:
            f.write(f'{wavname},{s_pesq},{s_stoi},{date["csig"]},{date["cbak"]},{date["covl"]},{date["pesq"]},{date["stoi"]},{date["ssnr"]}\n')
            
    

    def train(self):
        args = self.args
        model_name = self.model.module.__class__.__name__ if isinstance(self.model, nn.DataParallel) else self.model.__class__.__name__        
        # figname = f'{self.args.task}/{model_name}_{args.model}_{args.ssl_model}_{args.target}_{args.feature}_{args.size}_WS{args.weighted_sum}_FT{args.finetune_SSL}'
        figname = f'{self.args.task}/{model_name}_{args.model}_{args.ssl_model}_{args.target}_{args.lr}_{args.loss_fn}_{args.batch_size}_{args.feature}_{args.size}_WS{args.weighted_sum}_FT{args.finetune_SSL}'

        self.train_step = len(self.loader['train'])

        self.val_step = len(self.loader['val'])
        while self.epoch < self.epochs:
            self._train_epoch()
            self._val_epoch()

            self.writer.add_scalars(f'{figname}', {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(f'{figname}', {'val': self.val_loss},self.epoch)
            self.epoch += 1
            
    def test(self,model_path_test):
        # load model
        #后加的
        self.score_path_ = self.score_path
        self.score_path_=self.score_path_+'_'+model_path_test.split('_')[-1].split('.')[0]+'.csv'

        self.model.eval()
        checkpoint      = torch.load(model_path_test)
        self.model.load_state_dict(checkpoint['model'])
        noisy_paths        = get_filepaths(self.Test_path['noisy'])
        check_folder(self.score_path_)
        if os.path.exists(self.score_path_):
            os.remove(self.score_path_)
        with open(self.score_path_, 'a') as f:
            f.write('Filename,PESQ,STOI,MY_CSIG,MY_CBAK,MY_COVL,MY_PESQ,MY_STOI,MY_SSNR\n')
        for noisy_path in tqdm(noisy_paths):
            self.write_score(noisy_path,self.Test_path['clean'])

        data = pd.read_csv(self.score_path_)
        pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
        stoi_mean = data['STOI'].to_numpy().astype('float').mean()
        my_csig_mean = data['MY_CSIG'].to_numpy().astype('float').mean()
        my_cbak_mean = data['MY_CBAK'].to_numpy().astype('float').mean()
        my_covl_mean = data['MY_COVL'].to_numpy().astype('float').mean()
        my_pesq_mean = data['MY_PESQ'].to_numpy().astype('float').mean()
        my_stoi_mean = data['MY_STOI'].to_numpy().astype('float').mean()
        my_ssnr_mean = data['MY_SSNR'].to_numpy().astype('float').mean()

        with open(self.score_path_, 'a') as f:
            f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(my_csig_mean),str(my_cbak_mean),
                              str(my_covl_mean),str(my_pesq_mean),str(my_stoi_mean),str(my_ssnr_mean),))+'\n')
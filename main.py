

import torch.backends.cudnn as cudnn
import pandas as pd
import pdb
import os, argparse, torch, random, sys, torchaudio
from tensorboardX import SummaryWriter

# Import model and utility functions
from model.SE_module import SE_module_model
from model.Trainer import ModelTrainer
from utils.Load_model import load_pretrained_model, load_dataset
from utils.util import validate_directory

# Set audio backend to sox_io
torchaudio.set_audio_backend("sox_io")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training configurations for speech enhancement')
    
    # Add command-line arguments
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--data_dir', type=str, default='/home/zt/data/voicebank/data/data_16000') 
    parser.add_argument('--epochs', type=int, default=200) 
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--learning_rate', type=float, default=0.0001) 
    parser.add_argument('--loss_function', type=str, default='l1') 
    parser.add_argument('--feature', type=str, default='raw')
    parser.add_argument('--feature_type', type=str, default='fbak') 
    parser.add_argument('--optimizer', type=str, default='adam') 
    parser.add_argument('--model_name', type=str, default='FAodconv')   
    parser.add_argument('--ssl_model', type=str, default="wavlm") 
    parser.add_argument('--size', type=str, default='base') 
    parser.add_argument('--fine_tune_SSL', type=str, default='EF') 
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--target_type', type=str, default='IRM') 
    parser.add_argument('--task_type', type=str, default='SSL_SE')  
    parser.add_argument('--weighted_sum', action='store_true') 
    parser.add_argument('--checkpoint', type=str, default=None) 
    
    args = parser.parse_args()
    return args

def prepare_paths(args):
    # Paths for training, testing, checkpoints, models, and scores
    train_paths = {
        'noisy': f'{args.data_dir}/train_noisy/babble/0/',
        'clean': f'{args.data_dir}/train_clean/',
    }
    
    test_paths = {
        'noisy': f'{args.data_dir}/test_noisy/babble/0/',
        'clean': f'{args.data_dir}/test_clean/',
    }

    base_dir = '/home/zt/ssl/base_demo_BSSE-SE/exp'
    
    checkpoint_save_path = f"{base_dir}/checkpoint/{args.model_name}_{args.ssl_model}_{args.target_type}_epochs{args.epochs}_{args.optimizer}_{args.loss_function}_batch{args.batch_size}_lr{args.learning_rate}_{args.feature}_{args.size}_WS{args.weighted_sum}_FT{args.fine_tune_SSL}.pth.tar"
    
    model_save_path = f"{base_dir}/save_model/{args.model_name}_{args.ssl_model}_{args.target_type}_epochs{args.epochs}_{args.optimizer}_batch{args.batch_size}_lr{args.learning_rate}_{args.feature}_{args.size}_WS{args.weighted_sum}_FT{args.fine_tune_SSL}"
    
    score_save_path = f"{base_dir}/Result/{args.model_name}_{args.ssl_model}_{args.target_type}_epochs{args.epochs}_{args.optimizer}_{args.loss_function}_batch{args.batch_size}_lr{args.learning_rate}_{args.feature}_{args.size}_WS{args.weighted_sum}_FT{args.fine_tune_SSL}"
    
    return train_paths, test_paths, checkpoint_save_path, model_save_path, score_save_path

if __name__ == '__main__': 
    # Get current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Disable cuFFT plan cache to prevent memory issues
    torch.backends.cuda.cufft_plan_cache.max_size = 0

    # Parse arguments
    config = parse_arguments()
    print(f"Weighted sum setting: {config.weighted_sum}")
    
    # Declare paths for data and models
    train_data, test_data, ckpt_path, model_save_dir, score_dir = prepare_paths(config)
    
    # Set up tensorboard logger
    log_dir = '/home/zt/ssl/base_demo_BSSE-SE/exp/khhung/logs'
    writer = SummaryWriter(log_dir)

    # Initialize the model and load checkpoints
    model = SE_module_model(config)  # Use SE module model (contains SSL and BLSTM)
    model, start_epoch, best_model_loss, optimizer, loss_criteria, device = load_pretrained_model(config, model, ckpt_path, model_save_dir)
    
    # Load data
    data_loader = load_dataset(config, train_data)
    
    # Initialize trainer
    trainer = ModelTrainer(model, config.epochs, start_epoch, best_model_loss, optimizer, loss_criteria, device, data_loader, test_data, writer, model_save_dir, score_dir, config)
    
    try:
        if config.mode == 'train':
            trainer.train()
        
        saved_models_dir = '/home/zt/ssl/base_demo_BSSE-SE/exp/save_model'
        model_files = os.listdir(saved_models_dir)
        
        for filename in model_files:
            if model_save_dir.split('/')[7] in filename:
                test_model_path = os.path.join(saved_models_dir, filename)
                trainer.test(test_model_path)
                
    except KeyboardInterrupt:
        # Save the model in case of an interrupt
        model_state = {
            'epoch': start_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_model_loss
        }
        
        validate_directory(ckpt_path)
        torch.save(model_state, ckpt_path)
        print('Training interrupted. Model saved.')

        # Exit gracefully
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

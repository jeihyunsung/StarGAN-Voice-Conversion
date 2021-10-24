import os
import argparse
from data_loader import get_loader, TestDataset
from torch.backends import cudnn
from torch import cuda, device
from solver import Solver

def GPU_info():
    # GPU 할당 변경하기
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    curr_device = device(f'cuda:{GPU_NUM}' if cuda.is_available() else 'cpu')
    cuda.set_device(curr_device) # change allocation of current GPU
    print ('Current cuda device ', cuda.current_device()) # check

    # Additional Infos
    if device.type == 'cuda':
        print(cuda.get_device_name(GPU_NUM))
        print('Memory Usage:')
        print('Allocated:', round(cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
        print('Cached:   ', round(cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True
    GPU_info()

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    train_loader = get_loader(config.train_data_dir, config.batch_size, 'train', num_workers=config.num_workers)
    test_loader = TestDataset(config.test_data_dir, config.wav_dir, src_spk='p262', trg_spk='p272')

    # Solver for training and testing StarGAN.
    solver = Solver(train_loader, test_loader, config)

    if config.mode == 'train':    
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')
    parser.add_argument('--lambda_cls', type=float, default=10, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_div', type=float, default=10, help='weight for diversity loss')
    parser.add_argument('--lambda_sty', type=float, default=10, help='weight for style reconstruction loss')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--train_data_dir', type=str, default='/hdd_4T/hynsng/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='/hdd_4T/hynsng/mc/test')
    parser.add_argument('--wav_dir', type=str, default="/hdd_4T/hynsng/VCTK-Corpus/wav16")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--sample_dir', type=str, default='./samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--style_dim', type=int, default=64)

    # For Test
    parser.add_argument('--test_dir', type=str, default=None, help='WAV file directory for test')
    parser.add_argument('--test_save_dir', type=str, default='./', help='WAV file save directory for test')

    config = parser.parse_args()
    print(config)
    main(config)
import sklearn
from model import Generator
from model import Discriminator
from model import StyleEncoder
from model import MappingNetwork
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
from data_loader import speakers, spk2idx
from tqdm import tqdm
import soundfile as sf
import librosa
from utils import *

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # Model configurations.
        self.num_speakers = config.num_speakers
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_div = config.lambda_div
        self.lambda_sty = config.lambda_sty
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device : ", self.device, "cuda : ", torch.cuda.is_available())

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.latent_dim = config.latent_dim
        self.style_dim = config.style_dim

        # For Test
        self.test_dir = config.test_dir
        self.test_save_dir = config.test_save_dir

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(num_speakers=self.num_speakers)
        self.D = Discriminator(num_speakers=self.num_speakers)
        self.E = StyleEncoder(num_speakers=self.num_speakers)
        self.M = MappingNetwork(num_speakers=self.num_speakers)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.e_optimizer = torch.optim.Adam(self.E.parameters(), self.g_lr, [self.beta1, self.beta2]) # TODO - lr change
        self.m_optimizer = torch.optim.Adam(self.M.parameters(), self.g_lr, [self.beta1, self.beta2]) # TODO - lr change

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.E, 'E')
        self.print_network(self.M, 'M')

        self.G.to(self.device)
        self.D.to(self.device)
        self.E.to(self.device)
        self.M.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(resume_iters))
        M_path = os.path.join(self.model_save_dir, '{}-M.ckpt'.format(resume_iters))

        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))
        self.M.load_state_dict(torch.load(M_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.e_optimizer.zero_grad()
        self.m_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def adv_loss(self, logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    def r1_reg(self, d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return wav_padding(wav, sr=16000, frame_period=5, multiple = 4)  # TODO

    def train(self):
        """Train StarGAN."""
        # Set data loader.
        train_loader = self.train_loader

        data_iter = iter(train_loader)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]

        # Determine whether do copysynthesize when first do training-time conversion test.
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                mc_real, mc_real2, spk_label_org, spk_c_org = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_real, mc_real2, spk_label_org, spk_c_org = next(data_iter)

            mc_real = mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
            mc_real2 = mc_real2.unsqueeze_(1)

            # Generate target domain labels randomly.
            # spk_label_trg: int,   spk_c_trg:one-hot representation 
            spk_label_trg, spk_c_trg = self.sample_spk_c(mc_real.size(0)) 

            mc_real = mc_real.to(self.device)                         # Input mc.
            mc_real2 = mc_real2.to(self.device)                       # Input mc2.
            spk_label_org = spk_label_org.to(self.device)             # Original spk labels.
            spk_c_org = spk_c_org.to(self.device)                     # Original spk acc conditioning.
            spk_label_trg = spk_label_trg.to(self.device)             # Target spk labels for classification loss for G.
            spk_c_trg = spk_c_trg.to(self.device)                     # Target spk conditioning.

            z_trg = torch.randn(mc_real.size(0), self.latent_dim)
            z_trg = z_trg.unsqueeze_(1)
            z_trg = torch.FloatTensor(z_trg)
            z_trg = z_trg.to(self.device)

            z_trg2 = torch.randn(mc_real.size(0), self.latent_dim)
            z_trg2 = z_trg2.unsqueeze_(1)
            z_trg2 = torch.FloatTensor(z_trg2)
            z_trg2 = z_trg2.to(self.device)


            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real mc feats.
            mc_real.requires_grad_()
            out_src = self.D(mc_real, spk_label_org)
            d_loss_real = self.adv_loss(out_src, 1)
            d_loss_gp = self.r1_reg(out_src, mc_real)

            # Compute loss with fake mc feats.
            with torch.no_grad():
                if (i+1)%2 == 0:
                    style_vec = self.M(z_trg, spk_label_trg)
                else:
                    style_vec = self.E(mc_real, spk_label_trg)
            mc_fake = self.G(mc_real, style_vec)
            out_src = self.D(mc_fake, spk_label_trg)
            d_loss_fake = self.adv_loss(out_src, 0)

            # Compute loss for gradient penalty.
            # alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
            # x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
            # out_src = self.D(x_hat, spk_label_trg)
            # d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                if ((i+1)//self.n_critic)%2 == 0:
                    style_vec_trg = self.M(z_trg, spk_label_trg)
                else:
                    style_vec_trg = self.E(mc_real, spk_label_trg)

                mc_fake = self.G(mc_real, style_vec_trg)
                out_src = self.D(mc_fake, spk_label_trg)
                g_loss_fake = - torch.mean(out_src)

                # Target-to-original domain. (cycle consistency loss)
                if ((i+1)//self.n_critic)%2 == 0:
                    style_vec_org = self.M(z_trg, spk_label_org)
                else:
                    style_vec_org = self.E(mc_real, spk_label_org)

                mc_reconst = self.G(mc_fake, style_vec_org)
                g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                # Style reconstruction loss
                style_vec_pred = self.E(mc_fake, spk_label_trg)
                E_loss_sty = torch.mean(torch.abs(style_vec_pred-style_vec_trg))

                # Diversity sensitive loss
                if ((i+1)//self.n_critic)%2 == 0:
                    style_vec_div = self.M(z_trg2, spk_label_trg)
                else:
                    style_vec_div = self.E(mc_real2, spk_label_trg)
                mc_fake2 = self.G(mc_real, style_vec_div)
                E_loss_div = - torch.mean(torch.abs(mc_fake-mc_fake2.detach()))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_sty * E_loss_sty + self.lambda_div * E_loss_div
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                if ((i+1)//self.n_critic)%2 == 0:
                    loss['M/M_loss_sty'] = E_loss_sty.item()
                    loss['M/M_loss_div'] = E_loss_div.item()
                else:
                    loss['E/E_loss_sty'] = E_loss_sty.item()
                    loss['E/E_loss_div'] = E_loss_div.item()
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            if (i+1) % self.sample_step == 0:
                sampling_rate=16000
                num_mcep=36
                frame_period=5
                with torch.no_grad():
                    for idx, wav in tqdm(enumerate(test_wavs)):
                        wav_name = basename(test_wavfiles[idx])
                        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        f0_converted = pitch_conversion(f0=f0, 
                            mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src, 
                            mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                        
                        coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(self.device)

                        z_test = torch.FloatTensor(torch.randn(1, self.latent_dim)).unsqueeze_(1).to(self.device)
                        in_idx = torch.LongTensor(np.array([self.test_loader.spk_idx],dtype=np.int64)).to(self.device)
                        out = self.M(z_test, in_idx)
                        coded_sp_converted_norm = self.G(coded_sp_norm_tensor, out).data.cpu().numpy()
                        coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                                ap=ap, fs=sampling_rate, frame_period=frame_period)

                        sf.write(join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), wav_transformed, sampling_rate)
                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                            sf.write(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate)
                    cpsyn_flag = False

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(i+1))
                M_path = os.path.join(self.model_save_dir, '{}-M.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.E.state_dict(), E_path)
                torch.save(self.M.state_dict(), M_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def check_final_iter(self):
        max_iter = max([int(f.split('-')[0]) for f in os.listdir(self.model_save_dir) if os.path.isfile(join(self.model_save_dir, f))])
        return max_iter

    def get_demo_file_list(self):
        extension = self.test_dir.split('.')[-1]
        if extension == 'wav':
            demo_list = [self.test_dir]
        else:
            demo_list = [os.path.join(self.test_dir, f) for f in os.listdir(self.test_dir) if os.path.isfile(os.path.join(self.test_dir, f)) and f.split('.')[-1] == 'wav'] 
        return demo_list

    def test(self):

        # Test starGAN
        if self.test_dir:
            test_wavfiles = self.get_demo_file_list()
        else:
            test_wavfiles = self.test_loader.get_batch_test_data(batch_size=1)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]

        # if not self.test_save_dir:
        #     self.test_save_dir = './'

        if not self.resume_iters:
            self.resume_iters = self.check_final_iter()

        print("restore checkpoint at step %d ..."% self.resume_iters)
        self.restore_model(self.resume_iters)
            
        sampling_rate=16000
        num_mcep=36
        frame_period=5

        # Determine whether do copysynthesize when first do training-time conversion test.
        original_flag = True

        for idx, wav in tqdm(enumerate(test_wavs)):
            original_flag = True
            wav_name = basename(test_wavfiles[idx])

            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0, 
                mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src, 
                mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            
            coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(self.device)
            speakers_1 = ['p292', 'p293']
            for trg_spk in speakers_1:
                spk_idx =  spk2idx[trg_spk]
                z_test = torch.FloatTensor(torch.randn(1, self.latent_dim)).unsqueeze_(1).to(self.device)
                in_idx = torch.LongTensor(np.array([spk_idx],dtype=np.int64)).to(self.device)
                out = self.M(z_test, in_idx)
                coded_sp_converted_norm = self.G(coded_sp_norm_tensor, out).data.cpu().numpy()
                coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)

                sf.write(join(self.test_save_dir, 'fake'+'-'+wav_name.split('.')[0]+'-test-{}'.format(trg_spk)+'.wav'), wav_transformed, sampling_rate)
                if original_flag:
                    wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                ap=ap, fs=sampling_rate, frame_period=frame_period)
                    sf.write(join(self.test_save_dir, 'original-'+wav_name), wav_cpsyn, sampling_rate)
                    original_flag = False



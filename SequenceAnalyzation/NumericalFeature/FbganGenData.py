#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from utils.bio_utils import *
from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse
from utils.torch_utils import *
from utils.utils import *
from amp_predictor_pytorch import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from models import *



class WGAN_LangGP():
    def __init__(self, batch_size=64, lr=0.0001, num_epochs=150, seq_len = 156, data_dir='./data/dna_uniprot_under_50_reviewed.fasta', \
        run_name='test', hidden=512, d_steps = 10, max_examples=2000):
        self.preds_cutoff = 0.8
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = 1
        self.lamda = 10 #lambda
        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.sample_dir = './samples/' + run_name + "/"
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.charmap = {'P': 0, 'A': 1, 'T': 2, 'G': 3, 'C': 4}
        self.inv_charmap = ['P', 'A', 'T', 'G', 'C']
        self.build_model()

    def build_model(self):
        self.G = Generator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.D = Discriminator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        print(self.G)
        print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.analyzer = ACPClassifier() #from PYTORCH
        val_loss, val_acc = self.analyzer.evaluate_model()
        print("Val Acc:{}".format(val_acc))

        
    def load_model(self, directory = '', iteration=None):
        '''
            Load model parameters from most recent epoch
        '''
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        if iteration is None:
            print("Loading most recently saved...")
            G_file = max(list_G, key=os.path.getctime)
            D_file = max(list_D, key=os.path.getctime)
        else:
            G_file = "G_weights_{}.pth".format(iteration)
            D_file = "D_weights_{}.pth".format(iteration)
        epoch_found = int( (G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))
        return epoch_found
    
    def gen_data(self,n_batches,seed):
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        gen_seqs = []
        for i in range(n_batches):
            z = to_var(torch.randn(self.batch_size,128,generator=g_cpu))
            self.G.eval()
            torch_seqs = self.G(z)
            seqs = (torch_seqs.data).cpu().numpy()
            gen_seqs += [decode_one_seq(seq, self.inv_charmap) for seq in seqs]
        self.G.train()
        with open(self.sample_dir + 'fb_gen.txt','w',encoding='utf-8') as f:
            f.writelines([s + '\n' for s in gen_seqs])

def main():
    parser = argparse.ArgumentParser(description='FBGAN with AMP analyzer.')
    parser.add_argument("--run_name", default= "fbgan_amp_demo", help="Name for output files")
    parser.add_argument("--load_dir", default="./checkpoint/fbgan_amp_demo/", help="Load pretrained GAN checkpoints")
    args = parser.parse_args()
    model = WGAN_LangGP(run_name=args.run_name)
    # model.train_model(args.load_dir)
    n_batches = 128
    model.load_model(args.load_dir)
    model.gen_data(n_batches,12345)

if __name__ == '__main__':
    main()

# %%
#calc the frequences for samples generated in pre-train, trainning ,after trained
#compare them with real samples
#translate them into peptides
from cProfile import run

import importlib_resources
from utils.utils import *
import matplotlib.pyplot as plt
import utils.language_helpers as lh
from utils.bio_utils import *
import numpy as np


# %%
codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def countCorrectP(dna_seqs, verbose=False):
    global codon_table
    total = 0.0
    correct = 0.0
    for dna_seq in dna_seqs:
        p_seq = ""
        total += 1
        if dna_seq[0:3] != 'ATG':
            if verbose: print("Not valid gene (no ATG)")
            continue
        for i in range(3, len(dna_seq), 3):
            codon = dna_seq[i:i+3]
            try:
                aa = codon_table[codon]
                p_seq += aa
                if aa == '_': 
                    break
            except:
                if verbose: print("Error! Invalid Codon {} in {}".format(codon, dna_seq))
                break
        if len(p_seq) <= 2: #needs to have stop codon and be of length greater than 2
            if verbose: print("Error! Protein too short.")
        elif p_seq[-1] != '_':
            if verbose: print("Error! No valid stop codon.")
        else:
            correct+=1
    return correct/total


# %%
def plo(aim_list,labellist,xlabel,ylim):
    for i in range(len(labellist)):
        plt.plot(aim_list[i],labellist[i])
    plt.xlabel(xlabel=xlabel)
    plt.ylim(bottom=ylim)
    plt.legend(loc='lower right')

def is_PInDNA(line):
    for i in range(len(line)):
        if i+1< len(line) and line[i]=="P" \
        and line[i+1].isalpha() and line[i+1]!='P':
            return True
    return False

def cutline(line):
    mark = 0
    for i in range(len(line)):
        if line[i] == "P" or i == len(line) - 1 or line[i].isalpha()==False:
            mark = i
            break
    return line[0:mark]
    
def divideline(lines):
    dst = []
    line = ''
    for char in lines[0]:
        line += char
        if len(line) == 156:
            dst.append(line)
            line = ''
    return dst

def CalcFreq(path):
    with open(path,encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
        # if len(lines) == 1:
        #     lines = divideline(lines)
        cnt_total = 0.0
        cnt_PInDNA = 0.0
        freq_A = 0.0
        freq_T = 0.0
        freq_G = 0.0
        freq_C = 0.0
        freq_P = 0.0
        mean_len = 0.0
        mean_correctP = countCorrectP(lines)
        for line in lines:
            if len(line) == 0:continue
            cnt_total+=1
            freq_P += (float(line.count('P'))/len(line))
            if is_PInDNA(line):
                cnt_PInDNA+=1
            line  = cutline(line)
            len_dna = len(line)
            if len_dna == 0:
                continue
            mean_len += len_dna
            freq_A += float(line.count('A'))/len_dna
            freq_T += (float(line.count('T')))/len_dna
            freq_G += (float(line.count('G')))/len_dna
            freq_C += (float(line.count('C')))/len_dna
        freq_A /= cnt_total
        freq_T /= cnt_total
        freq_G /= cnt_total
        freq_C /= cnt_total
        freq_P /= cnt_total
        mean_len/= cnt_total
        cnt_PInDNA/=cnt_total
        plines = geneToProtein(lines,verbose=False)
        mean_Lp = 0.0
        for pline in plines:
            mean_Lp+= len(pline)
        if len(plines) == 0 :
            mean_Lp = 0
        else:
            mean_Lp/= len(plines)
        return freq_A,freq_T,freq_G,freq_C,freq_P,mean_len,cnt_PInDNA,mean_correctP,mean_Lp
    
def divide(path):
    content = []
    with open(path,encoding='utf-8') as f:
        content = f.read()
    lines = []
    line = ''
    # print(len(content))
    # print(content[0])
    for char in content:
        line += char
        if len(line) == 156:
            lines.append(line)
            line = ''
    
    with open(path,'w',encoding='utf-8') as f:
        f.writelines([s + '\n' for j, s in enumerate(lines)])
        

# %%
# for i in range(12400):
#     if i % 100 == 99:
#         path = './samples/realProt_50aa/sampled_val_{}.txt'.format(i)
#         divide(path)

# %%
def running(run_name,sample_name):
    mean_freq_A = []
    mean_freq_T = []
    mean_freq_G = []
    mean_freq_C = []
    mean_freq_P = []
    mean_Length = []
    mean_Lps = []
    cnt_PInS = []
    mean_correctPs = []
    for i in range(31100):
        if i % 100 == 99:
            # path = './samples/realProt_50aa/sampled_{}.txt'.format(i)
            path = './samples/realProt_50aa/sampled_{}.txt'.format(i)
            # path = './samples/' + sample_name + '/sampled_{}.txt'.format(i)
            freq_A,freq_T,freq_G,freq_C,freq_P,mean_len,cnt_PInDNA,mean_correctP,meanLp = CalcFreq(path)
            mean_freq_A.append(freq_A)
            mean_freq_T.append(freq_T)
            mean_freq_G.append(freq_G)
            mean_freq_C.append(freq_C)
            mean_freq_P.append(freq_P)
            mean_Length.append(mean_len)
            cnt_PInS.append(cnt_PInDNA)
            mean_correctPs.append(mean_correctP)
            mean_Lps.append(meanLp)


    # %%
    mean_fb_freq_A = []
    mean_fb_freq_T = []
    mean_fb_freq_G = []
    mean_fb_freq_C = []
    mean_fb_freq_P = []
    mean_fb_Length = []
    mean_fb_correctPs = []
    fb_cnt_PInS = []
    mean_fb_Lps = []
    for i in range(150):
            # path = './samples/fbgan_amp_demo/sampled_{}_preds.txt'.format(i+1)
            path = './samples/' + sample_name + '/sample_val_{}.txt'.format(i+1)
            freq_A,freq_T,freq_G,freq_C,freq_P,mean_len,cnt_PInDNA,mean_fb_correctP,meanLp = CalcFreq(path)
            mean_fb_freq_A.append(freq_A)
            mean_fb_freq_T.append(freq_T)
            mean_fb_freq_G.append(freq_G)
            mean_fb_freq_C.append(freq_C)
            mean_fb_freq_P.append(freq_P)
            mean_fb_Length.append(mean_len)
            fb_cnt_PInS.append(cnt_PInDNA)
            mean_fb_correctPs.append(mean_fb_correctP)
            mean_fb_Lps.append(meanLp)


    # %%
    real_datas,_,__ = lh.load_dataset(max_length=156,max_n_examples=2048,data_dir='./data/AMP_dataset.fa')
    real_lines = []
    for data in real_datas:
            line = ''
            for char in data:
                    if char.isalpha():
                            line += char
            real_lines.append(line)
    # print(real_lines)
    store_path = './samples_real/realsamplesamp.txt'
    with open(store_path,mode='w+',encoding='utf-8') as f:
            content = ''
            for line in real_lines:
                    content+= line
                    content+='\n'
            f.write(content)


    # %%
    mean_real_freq_A = []
    mean_real_freq_T = []
    mean_real_freq_G = []
    mean_real_freq_C = []
    mean_real_freq_P = []
    mean_real_Length = []
    real_cnt_PInS = []
    mean_real_Lps = []

    freq_A,freq_T,freq_G,freq_C,freq_P,mean_len,cnt_PInDNA,_,meanLp = CalcFreq(store_path)
    mean_real_freq_A = [freq_A] * len(mean_fb_freq_A)
    mean_real_freq_T = [freq_T] * len(mean_fb_freq_A)
    mean_real_freq_G = [freq_G] * len(mean_fb_freq_A)
    mean_real_freq_C = [freq_C] * len(mean_fb_freq_A)
    mean_real_freq_P = [freq_P] * len(mean_fb_freq_A)
    mean_real_Length = [mean_len] * len(mean_fb_freq_A)
    real_cnt_PInS = [cnt_PInDNA] * len(mean_fb_freq_A)
    mean_real_Lps = [meanLp] * len(mean_fb_freq_A)


    # %%
    import matplotlib as mpl
    import os
    save_path = './' + run_name
    if not os.path.exists(save_path): os.makedirs(save_path)

    mpl.rcParams.update({'font.size': 20})

    plt.figure(figsize=(20,10))
    plt.plot(mean_Length, label='mean len in pre-train')
    plt.plot(mean_fb_Length, label='mean len in feedback train')
    plt.plot(mean_real_Length,label='mean len of real samples')

    plt.title('Mean length of nucleotides chains per batch')
    plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.grid()

    plt.legend(loc='lower right')
    plt.savefig(save_path + '/MeanLength.png')

    plt.figure(figsize=(20,10))
    plt.plot(mean_correctPs, label='freq of genes with correct structure in pre-train')
    plt.plot(mean_fb_correctPs, label='freq of genes with correct structure in feedback train')

    plt.title('frequence of genes with correct structure')
    plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.grid()

    plt.legend(loc='lower right')
    plt.savefig(save_path + '/CorrectGene.png')

    plt.figure(figsize=(20,10))
    plt.plot(cnt_PInS, label='Freq of seqs in pre-train')
    plt.plot(fb_cnt_PInS, label='Freq of seqs in feedback train')
    plt.plot(real_cnt_PInS, label= 'Freq of real samples')
    plt.title('Frequences of sequences that has multiple nucleotides chains')
    plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    plt.grid()

    plt.savefig(save_path + '/MultipleNC.png')

    plt.figure(figsize=(20,10))

    plt.plot(mean_Lps, label='Mean length of generated protein in Pre-train')
    plt.plot(mean_fb_Lps, label='Mean length of generated protein in feedback train')
    plt.plot(mean_real_Lps, label='Mean length of real protein')

    plt.title('Mean length of protein')
    plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.grid()

    plt.legend(loc='lower right')
    plt.savefig(save_path + '/MeanLenP.png')

    # %%
    plt.figure(figsize=(20,13))

    plt.plot(mean_fb_freq_A, label='freq_A in feedback train',linewidth=3)
    plt.plot(mean_fb_freq_T, label='freq_T in feedback train',linewidth=3)
    plt.plot(mean_fb_freq_G, label='freq_G in feedback train',linewidth=3)
    plt.plot(mean_fb_freq_C, label='freq_C in feedback train',linewidth=3)
    plt.plot(mean_real_freq_A, label='freq_A of real samples',linewidth=3)
    plt.plot(mean_real_freq_T, label='freq_T of real samples',linewidth=3)
    plt.plot(mean_real_freq_G, label='freq_G of real samples',linewidth=3)
    plt.plot(mean_real_freq_C, label='freq_C of real samples',linewidth=3)

    plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.grid()
    plt.title('Percentage of nucleotides per chain in feedback train')
    plt.legend(loc='lower left')

    plt.savefig(save_path + '/NDistribution_fb.png')

    plt.figure(figsize=(40,13))
    plt.plot(mean_freq_A, label='freq_A in pre-train',linewidth=2)
    plt.plot(mean_freq_T, label='freq_T in pre-train',linewidth=2)
    plt.plot(mean_freq_G, label='freq_G in pre-train',linewidth=2)
    plt.plot(mean_freq_C, label='freq_C in pre-train',linewidth=2)
    plt.plot(mean_real_freq_A[0:len(mean_freq_A)], label='freq_A of real samples',linewidth=2)
    plt.plot(mean_real_freq_T[0:len(mean_freq_A)], label='freq_T of real samples',linewidth=2)
    plt.plot(mean_real_freq_G[0:len(mean_freq_A)], label='freq_G of real samples',linewidth=2)
    plt.plot(mean_real_freq_C[0:len(mean_freq_A)], label='freq_C of real samples',linewidth=2)
    plt.grid()

    plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.title('Percentage of nucleotides per chain in pre-train')
    plt.legend(loc='lower right')
    plt.savefig(save_path + '/NDistribution_pre.png')


def main():
    parser = argparse.ArgumentParser(
        description='Freq calculation'
    )
    parser.add_argument("--run_name", default='AMP', help="Name for pic")
    parser.add_argument("--sample_name", default='fbgan_amp_demo', help="Name of samples")

    args = parser.parse_args()
    running(args.run_name,args.sample_name)

if __name__ == '__main__':
    main()

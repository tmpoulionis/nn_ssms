# Credits to https://github.com/HazyResearch/hyena-dna/blob/main/src/dataloaders/datasets/hg38_dataset.py
# Below is a modified version of their code

import os
import gzip
import shutil
from pathlib import Path
from pyfaidx import Fasta
import pandas as pd
import torch
from random import randrange, random
import numpy as np
from urllib.request import urlretrieve
from tqdm import tqdm
from dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
"""

Dataset for sampling arbitrary intervals from the human genome.

"""

# Utility classes
class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        # max_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        pad_interval = False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval        

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            # remove tail end, might be gibberish code
            # truncate_len = int(len(self.seqs[chr_name]) * 0.9)
            # self.chr_lens[chr_name] = truncate_len
            self.chr_lens[chr_name] = len(self.seqs[chr_name])


    def __call__(self, chr_name, start, end, max_length, return_augs = False):
        """
        max_length passed from dataset, not from init
        """
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        # chromosome_length = len(chromosome)
        chromosome_length = self.chr_lens[chr_name]

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        # checks if not enough sequence to fill up the start to end
        if interval_length < max_length:
            extra_seq = max_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        # Added support!  need to allow shorter seqs
        if interval_length > max_length:
            end = start + max_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq

class HG38Dataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
    '''

    # URLs for the dataset files
    FASTA_URL = "https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz"
    BED_URL = "https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed"
    chars = ['A', 'T', 'C', 'G', 'N', 'a', 't', 'c', 'g', 'n', '.']
    
    def __init__(
        self,
        subset,
        bed_file=None,
        fasta_file=None,
        max_length=None,
        root='./data/hg38',
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        download=True
    ):

        self.root = Path(root)
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = CharacterTokenizer(
            characters=self.chars,
            model_max_length=max_length
        )
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval         


        # Set default file paths if not provided
        if bed_file is None:
            bed_file = self.root / 'human-sequences.bed'
        if fasta_file is None:
            fasta_file = self.root / 'hg38.ml.fa'
            
        self.bed_file = Path(bed_file)
        self.fasta_file = Path(fasta_file)

        if download:
            self._download()
            
        # Verify files exist
        if not self.bed_file.exists():
            raise FileNotFoundError(
                f"BED file not found at {self.bed_file}. "
                "Set download=True to download automatically."
            )
        if not self.fasta_file.exists():
            raise FileNotFoundError(
                f"FASTA file not found at {self.fasta_file}. "
                "Set download=True to download automatically."
            )
            
        # read bed file
        df_raw = pd.read_csv(str(self.bed_file), sep = '\t', names=['chr_name', 'start', 'end', 'subset'])
        # select only subset df
        self.df = df_raw[df_raw['subset'] == subset]

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            # max_length = max_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug,
            pad_interval = pad_interval,
        )

    def _download(self):
        """Download and extract dataset files if they don't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download BED file
        if not self.bed_file.exists():
            self._download_file(self.BED_URL, self.bed_file)
            
        # Download and extract FASTA file
        if not self.fasta_file.exists():
            fasta_gz = self.root / 'hg38.ml.fa.gz'
            if not fasta_gz.exists():
                self._download_file(self.FASTA_URL, fasta_gz)
                self._extract_gzip(fasta_gz, self.fasta_file)
    
    @staticmethod
    def _download_file(url, output_path):
        """Download a file with progress bar."""
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urlretrieve(url, filename=output_path, reporthook=t.update_to)
            
    @staticmethod
    def _extract_gzip(gz_path, output_path):
        """Extract a gzip file."""
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        row = self.df.iloc[idx]
        # row = (chr, start, end, subset)
        chr_name, start, end = row.iloc[:3]

        seq = self.fasta(chr_name, start, end, max_length=self.max_length, return_augs=self.return_augs)

        if self.tokenizer_name == 'char':

            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        target = seq[1:].clone()  # offset by 1, includes eos
        
        return data, target
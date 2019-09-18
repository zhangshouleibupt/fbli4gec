import random
import difflib
import logging
from torch.utils.data import dataset
logger = logging.getLogger(__name__)
def read_token_lines_from_file(file):
    with open(file,'r',encoding='utf8') as f:
        lines = f.readlines()
    token_lines = [line.rstrip('\n').lstrip(' ') for line in lines]
    return token_lines

def compare_len(t):
    src,tgt = t[0].split(' '),t[1].split(' ')
    if len(src) == len(tgt):
        return 0
    elif len(src) < len(tgt):
        return -1
    else:
        return 1
#the none definitly accurate to label which word is wrong

def label_which_word_wrong(s1,s2):
    s1,s2 = s1.split(' '),s2.split(' ')
    word_compare_tuple_list = []
    label_list = []
    if len(s1) == len(s2):
        for w1,w2 in zip(s1,s2):
            if w1 == w2:
                label_list.append('s')
            else:
                label_list.append('r')
                word_compare_tuple_list.append((w1,w2))
    else:
        matcher = difflib.SequenceMatcher(None,s1,s2)
        label_list = ['s'] * len(s1)
        for tag,i1,i2,j1,j2 in matcher.get_opcodes():
            if tag == 'replace':
                label_list[i1:i2] = ['r'] * (i2-i1)
                word_compare_tuple_list += list(zip(s1[i1:i2],s2[i1:i2]))
            elif tag == 'insert':
                label_list[i1] = 'i'
                word_compare_tuple_list += list(zip([s1[i1]]*(j2-j1),s2[j1:j2]))
            elif tag == 'delete':
                label_list[i1:i2] = ['d'] * (i2 - i1)
                word_compare_tuple_list += list(zip(s1[i1:i2],['[DELETE]']*(i2-i1)))
    return label_list,word_compare_tuple_list
#raw sentence pair just source to target 
def load_data_into_parallel(src_file,trg_file):
    #return format:
    #[(src1,tgt1),...]
    src_lines = read_token_lines_from_file(src_file)
    trg_lines = read_token_lines_from_file(trg_file)
    assert len(src_lines) == len(trg_lines)
    return list(zip(src_lines,trg_lines))

class LanguageDataset(dataset.Dataset):
    r"""
    dataset for sequence to sequence,
    """
    def __init__(self,langs,token_method):
        super(LanguageDataset,self).__init__()
        self.lang_to_token_method_map = {
            'BPE' : self.bpe_method,
            'just_map' : self.just_mappping_method,
        }
        self.langs = langs
    def __getitem__(self,index):
        return self.langs[index]
    def __len__(self):
        return len(self.langs)

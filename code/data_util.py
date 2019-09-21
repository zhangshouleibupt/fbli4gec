import random
import difflib
import logging
from torch.utils.data import dataset
import fairseq as fs
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
    create this dataset for FairDataset used
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

class PaddedTensorLanguageDataset(dataset.Dataset):        
    r"""create the indexed format for the retraing stage
        return format: (padded_src_tensor,padded_trg_tensor,pad_mask)
    """

    def __init__(self,langs_pairs,reversed=False):
        super(PaddedTensorLanguageDataset,self).__init__()
        self.langs_paris = langs_pairs
        self.src_langs,self.trg_langs = list(zip(*self.langs_pairs))
        #cause the backboost tainning algorithm need this correted-to-error
        # pair format,we need reverse the pairs 
        if reversed:
            self.src_langs,self.trg_langs = self.trg_langs,self.src_langs
        self.dictionary = fs.data.Dictionay().load(config['word_dict'])
        self.max_len = config['max_len']
        self.pad_left = pad_left

    def __getitem__(self,index):
        src,trg = self.src_langs[index],self.trg_langs[index]
        return self._pad_one_pair(src,trg)

    def __len__(self):
        return len(self.src_langs)

    def is_pad_left(self):
        return self.pad_left

    def _pad_one_pair(self,src,trg):

        src, trg = src.split(' '), trg.split(' ')
        src_len, trg_len = len(src), len(trg)
        src_need_padded_len = self.max_len - src_len - 1 
        trg_need_padded_len = self.max_len - trg_len - 1
        src_idxs = [self.dictionary.index(token) for token in src]
        trg_idxs = [self.dictionary.index(token) for token in trg]
        src_idxs = src_idxs + [self.dictionary.eos()]
        trg_idxs = [self.dictionary.sos()] + trg_idxs
        #source padded left condition 
        if self.pad_left:
            src_idxs = [self.dictionary.pad()] * src_need_padded_len + src_idxs
            src_mask = [0] * src_need_padded_len + [1] * src_len
        else:
            src_idxs = src_idxs + [self.dictionary.pad()] * src_need_padded_len
            src_mask = [1] * src_len + [0] * src_need_padded_len

        trg_idxs = trg_idxs + [self.dictionary.pad()] * trg_need_padded_len
        trg_mask = [1] * trg_len + [0] * trg_need_padded_len
        src_idxs_tensor = torch.tensor(src_idxs,dtype=torch.long)
        trg_idxs_tensor = torch.tensor(trg_idxs,dtypr=torch.long)
        src_mask = torch.tensor(src_mask,dtype=torch.float32)
        trg_mask = torch.tensor(trg_mask,dtype=torch.float32)
        return (src_idx_tensor,trg_idxs_tensor,src_mask,trg_mask)

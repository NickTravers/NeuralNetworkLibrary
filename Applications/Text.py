# Text.py
from General.Core import *
from General.Layers import *
from General.Learner import *
from General.LossesMetrics import *
from General.Optimizer import *

# This file Text.py contains a collection of functions and classes 
# for use in some problems related to natural language processing.
# Specificallly, language modeling and text classification. 

# OUTLINE
# Section (1) - Tokenization and Numericalization
# Section (2) - Datasets and Data Objects
# Section (3) - Pytorch Models
# Section (4) - Loss Functions and Metrics


# SECTION (1) - TOKENIZATION AND NUMERICALIZATION

# NOTE: 
# The Tokenizer class below is modified only slightly from the fastai Tokenizer. 
# Specifically, the staticmethods 'tokenize' and 'tokenize_mp' have been removed
# and are now stand alone functions defined outside of the class. It is beneficial
# to use exactly the same tokenization scheme as fastai, because our language model
# is initialized with pre-trained token embedding vectors from a fastai model.  

class Tokenizer():
    """Tokenizer Class, wraps the Spacy Tokenizer with a little bit of extra pre-processing. 
       The main function is proc_text, which tokenizes a single text (i.e. string) s."""
    
    def __init__(self, lang='en'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.tok = spacy.load(lang)
        for w in ('<eos>','<bos>','<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_br(self,x): return self.re_br.sub("\n", x)

    def spacy_tok(self,x):
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP = ' t_up '
        res = []
        prev='.'
        re_word = re.compile('\w')
        re_nonsp = re.compile('\S')
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2)) else [s.lower()])
        return ''.join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

def tokenize(ss):
    "Tokenizes a list of texts ss. Each text s in ss is a string."
    print(' ', end='', flush=True)
    tok, ss_tokenized = Tokenizer(), []
    for s in PBar(ss): 
        ss_tokenized.append(tok.proc_text(s))
    return ss_tokenized

def tokenize_mp(ss, ncpus = os.cpu_count() - 2):
    "Tokenizes a list of texts ss using multi-processing. Each text s in ss is a string."
    clear_output()
    print('Tokenizing ...')
    n, m = len(ss), int(np.ceil(len(ss)/ncpus))
    ss_split = [ss[i:min(i+m,n)] for i in range(0, n, m)]

    with ProcessPoolExecutor(ncpus) as executor:
        return sum(executor.map(tokenize, ss_split), [])
    
def numericalize(ss, max_vocab=60000, min_freq=6, stoi=None):
    """
    Numericalizes a list of pre-tokenized texts ss.

    Arguments:
    ss: list of pre-tokenized texts (output of tokenize_mp function)
    stoi: If given, is a dictionary mapping string to int form of tokens.
          Should have stoi['_unk_'] = 0, default value of 0 is used for missing keys.
    max_vocab: If stoi not given, is the maximum number of tokens to use.
    min_freq: If stoi not given, any token occurring < min_freq times is treated as unknown (or '_unk_').

    Retuns: ss_numeric, stoi
    """
    print('Numericalizing ...')
    print('min freq =', min_freq, ', max_vocab =', max_vocab)
    
    if stoi is None:
        ss_joined = [tok for s in ss for tok in s]
        token_counts = collections.Counter(ss_joined).most_common(max_vocab)
        tokens = [tok for tok,count in token_counts if count >= min_freq]
        special_tokens = ['_unk_', '_pad_', '_bos_', '_eos_']
        tokens = special_tokens + tokens
        stoi = {tok:i for i,tok in enumerate(tokens)}
        
    stoi_copy = collections.defaultdict(lambda:0, stoi)
    ss_numeric = [[stoi_copy[tok] for tok in s] for s in ss]
    print('Done, vocab_size = ', len(stoi))
    return ss_numeric, stoi


# SECTION (2) - DATASETS AND DATA OBJECTS

class TextDataset(object):
    "General purpose text dataset, for both language modeling and text classification."

    def __init__(self, texts, labels, stoi=None, reverse=False):
        
        """
        Arguments:
        texts: A list of texts, each text in texts is a single string.
        labels: A list of integer labels for the texts. If no labels are needed/present 
                (e.g. for a test dataset or in language modeling), then set all labels to 0.
        stoi: A dictionary used to map string form of tokens to int form (or None).
        reverse: If True, list of tokens corresponding to each text is put in reverse order.
        """
        
        texts = tokenize_mp(texts)
        self.texts, self.stoi = numericalize(texts, stoi=stoi)
        self.texts = pd.Series(self.texts)
        if reverse: self.texts = pd.Series([list(reversed(t)) for t in self.texts])
        self.num_tokens = sum(len(t) for t in self.texts)
        
        unique_labels = sorted(list(set(labels)))
        self.label_dict = {lab:i for i,lab in enumerate(unique_labels)}
        self.labels = pd.Series([self.label_dict[lab] for lab in labels])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        return self.texts.iloc[idx], self.labels.iloc[idx]

    def split_train_val(self):
        
        all_texts = copy.deepcopy(self.texts)
        all_labels = copy.deepcopy(self.labels)
        all_idxs = list(range(len(self.texts)))
        
        train_ds, val_ds = self, copy.deepcopy(self)
        train_idxs, val_idxs = SplitTrainVal(all_idxs)
        
        train_ds.texts = all_texts[train_idxs]
        train_ds.texts.index = range(len(train_ds.texts))
        train_ds.labels = all_labels[train_idxs]
        train_ds.labels.index = range(len(train_ds.labels))
        train_ds.num_tokens = sum(len(t) for t in train_ds.texts)
        
        val_ds.texts = all_texts[val_idxs]
        val_ds.texts.index = range(len(val_ds.texts))
        val_ds.labels = all_labels[val_idxs]
        val_ds.labels.index = range(len(val_ds.labels))
        val_ds.num_tokens = sum(len(t) for t in val_ds.texts)
 
        return train_ds, val_ds

    @classmethod
    def from_csv(cls, csv_file, text_col, label_col=None, stoi=None, reverse=False):
        """ Constructs a TextDataset object from a .csv file.
            Each line of the csv contains a single text, and possibly also a label. """

        df = pd.read_csv(csv_file)   
        if label_col: return cls(list(df[text_col]), list(df[label_col]), stoi, reverse)
        else: return cls(list(df[text_col]), [0]*len(df), stoi, reverse)

    @classmethod
    def from_text_files(cls, folder, labels, stoi=None, reverse=False):
        """ Constructs a TextDataset object from a folder containing text files 
            (possibly in labeled subfolders). 
        
        'labels' can be either None, 'All', or a list.         
        * If None, 'folder' should contain .txt files. 
        * If 'All', 'folder' should contain only sub-folders with the 
          label names. Each subfolder should contain .txt files.
        * If a list, for each label in labels there should be a sub-folder of
          'folder' containing .txt files. But other things may also be in 'folder'.
        """
        
        folder = correct_foldername(folder)

        if labels is None:

            texts = []
            filenames = os.listdir(folder)
            filenames = [fn for fn in filenames if fn[-4:] == '.txt']
            for fn in filenames:
                f = open(folder + fn, 'r')
                texts.append(f.read())
            texts_labels = [0]*len(texts)

        if labels is not None:
            
            if type(labels)==str: 
                labels = os.listdir(folder)
            labels.sort()
            texts, texts_labels = [],[]

            for lab in labels:
                filenames = os.listdir(folder + lab)
                filenames = [fn for fn in filenames if fn[-4:] == '.txt']
                texts_labels += [lab] * len(filenames)
                for fn in filenames:
                    f = open((folder + lab + '/' + fn), 'r')
                    texts.append(f.read())

        return cls(texts, texts_labels, stoi, reverse)

class LanguageModelDataLoader(object):
    """Language Model Dataloader for a TextDataset. 
       
       All texts in the dataset ds are concatenated into 1 big text, and then concatenated 
       text is split into bs consecutive chunks. This gives a new reshaped object of shape 
       (bs x seqlen) where seqlen is roughly equal to (total number tokens in ds)/bs. 
       Batches of shape (bs x bptt), are then yielded sequentially from reshaped object. """

    def __init__(self, ds, bs, bptt, random=True):
        
        """
        Arguments:
        ds: A dataset of class TextDataset.
        bs: batch size.
        bptt: Stands for 'back prop through time', batches of shape (bs x bptt) 
              tokens are generated by dataloader.
        random: If True, order of texts is randomized before concatenation, and also the 
                value of bptt is randomly perturbed from its default value for each batch. 
                Set to True for train dataset, and False for val or test dataset.
        """

        self.bs, self.bptt, self.random = bs, bptt, random
        self.texts, self.ntexts = ds.texts, len(ds.texts)
        self.seqlen = (ds.num_tokens//bs) - 1
        self.ntoks = bs * (self.seqlen + 1)
        self.concat_texts()
        self.set_batch_lengths()
        
    def concat_texts(self):
        idxs = list(range(self.ntexts))
        if self.random: np.random.shuffle(idxs)        
        self.combined_text = np.array([tok for i in range(self.ntexts) for tok in self.texts.iloc[idxs[i]]])[:self.ntoks]
        self.combined_text = self.combined_text.reshape(self.bs, self.seqlen+1)
    
    def set_batch_lengths(self):
        self.batch_lengths = []
        i, ntoks_used = 0, 0       
        while ntoks_used < self.seqlen:
            bptt = self.bptt
            if self.random and i>0 and np.random.random() < 0.05: bptt = bptt//2
            if self.random and i>0: bptt = bptt - np.random.randint(0,10)
            ntoks_batch = min(self.seqlen - ntoks_used, bptt)
            ntoks_used += ntoks_batch; i += 1
            self.batch_lengths.append(ntoks_batch)
    
    def __len__(self):
        return len(self.batch_lengths)
        
    def __iter__(self):
        
        # Loop through yielding batches of size (bs x bl) for bl in self.batch_lengths
        ntoks_used = 0
        for bl in self.batch_lengths:
            batch_x = TEN(self.combined_text[:, ntoks_used: ntoks_used + bl])
            batch_y = TEN(self.combined_text[:, ntoks_used + 1: ntoks_used + bl + 1])
            ntoks_used += bl
            yield (batch_x, batch_y)
            
        # Re-concatenate texts in new random order if self.random == True
        if self.random == True: self.concat_texts()

class LanguageModelDataObj(object):
    """ Class for a language model data object encompassing the datasets and 
        corresponding dataloaders for train, validation, and (optionally) test data, 
        along with a bit of extra information."""
    
    def __init__(self, train_ds, val_ds, test_ds, bs, bptt):

        self.bs, self.bptt, self.stoi, self.target_type = bs, bptt, train_ds.stoi, 'lang_model'
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds

        self.train_dl = LanguageModelDataLoader(train_ds, bs, bptt, True)
        self.val_dl = LanguageModelDataLoader(val_ds, bs, bptt, False)
        if test_ds: self.test_dl = LanguageModelDataLoader(test_ds, bs, bptt, False)

    @classmethod
    def from_csv(cls, bs, bptt, csv_train, csv_val=None, csv_test=None, text_col='text', reverse=False):

        train_ds = TextDataset.from_csv(csv_train, text_col, label_col=None, stoi=None, reverse=reverse)
        stoi = train_ds.stoi

        if csv_val: val_ds = TextDataset.from_csv(csv_val, text_col, label_col=None, stoi=stoi, reverse=reverse)
        else: train_ds, val_ds = train_ds.split_train_val()

        if csv_test: test_ds = TextDataset.from_csv(csv_test, text_col, label_col=None, stoi=stoi, reverse=reverse)
        else: test_ds = None

        return cls(train_ds, val_ds, test_ds, bs, bptt)

    @classmethod
    def from_folders(cls, bs, bptt, labels, train, val=None, test=None, reverse=False):

        train_ds = TextDataset.from_text_files(train, labels, stoi=None, reverse=reverse)
        stoi = train_ds.stoi

        if val: val_ds= TextDataset.from_text_files(val, labels, stoi=stoi, reverse=reverse)
        else: train_ds, val_ds = train_ds.split_train_val()

        if test: test_ds = TextDataset.from_text_files(test, labels, stoi=stoi, reverse=reverse)
        else: test_ds = None

        return cls(train_ds, val_ds, test_ds, bs, bptt)

class TextLengthSampler(Sampler):

    """ Sampler for a TextDataset based on length of texts, for use with text classification problems.
        Yields batches of text idxs, such that all texts in each batch have approximately same length."""

    def __init__(self, ds, bs, bpg=10, random=False):
        """ 
        Arguments:
        ds: A dataset of class TextDatset.
        bs: The batch size. 
        bpg: Number of batches per group.
        random: If True, order of groups is randomized and texts within each group are also sampled in random order.
                If False, neither is random. So, all texts are sampled in order according to their length.
        """

        L = len(ds)
        perm = list(range(L))
        perm.sort(key = lambda i:len(ds.texts[i]), reverse=True)
        ds.texts, ds.labels, ds.perm = ds.texts[perm], ds.labels[perm], perm
        ds.texts.index, ds.labels.index = range(L), range(L)      

        group_sz = bs*bpg
        self.groups = [list(range(i,min(i+group_sz,L))) for i in range(0,L,group_sz)]
        self.bs, self.random = bs, random
        self.length = len([x for x in self])
        
    def __len__(self):
        return self.length

    def __iter__(self):

        if self.random:
            groups = self.groups[1:]
            np.random.shuffle(groups)
            self.groups = [self.groups[0]] + groups

        for g in self.groups:
            if self.random:
                np.random.shuffle(g)
            for i in range(0, len(g), self.bs):
                idxs = g[i: min(i+self.bs, len(g))]
                yield idxs

class TextLengthCollater(object):
    "Collater for batches sampled according to TextLengthSampler."

    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        texts, labels = [b[0] for b in batch], [b[1] for b in batch]
        m = max([len(t) for t in texts])
        for i in range(len(texts)):
            texts[i] = np.array( texts[i] + [self.pad_token]*(m-len(texts[i])) )

        return TEN(texts, GPU=False), TEN(labels, GPU=False)

class TextClassificationDataObj(object):
    """ Class for a text classification data object encompassing the datasets and 
        corresponding dataloaders for train, validation, and (optionally) test data, 
        along with a bit of extra information."""

    def __init__(self, train_ds, val_ds, test_ds, bs, bpg=10, num_workers=6):

        self.bs, self.stoi, self.target_type = bs, train_ds.stoi, 'text_classify'
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds

        collater = TextLengthCollater(self.stoi['_pad_'])
        sampler_train = TextLengthSampler(train_ds, bs, bpg, random=True)
        sampler_val = TextLengthSampler(val_ds, bs, bpg, random=False)
        if test_ds: sampler_test = TextLengthSampler(test_ds, bs, bpg, random=False)

        self.train_dl = DataLoader(train_ds, collate_fn=collater, batch_sampler=sampler_train, num_workers=num_workers)
        self.val_dl = DataLoader(val_ds, collate_fn=collater, batch_sampler=sampler_val, num_workers=num_workers)
        if test_ds:
            self.test_dl = DataLoader(test_ds, collate_fn=collater, batch_sampler=sampler_test, num_workers=num_workers)

    @classmethod
    def from_csv(cls, bs, csv_train, csv_val=None, csv_test=None, text_col='text', label_col='label', 
                 reverse=False, stoi=None):

        train_ds = TextDataset.from_csv(csv_train, text_col, label_col, stoi=stoi, reverse=reverse)
        stoi = train_ds.stoi

        if csv_val: val_ds = TextDataset.from_csv(csv_val, text_col, label_col, stoi=stoi, reverse=reverse)
        else: train_ds, val_ds = train_ds.split_train_val()

        if csv_test: test_ds = TextDataset.from_csv(csv_test, text_col, label_col, stoi=stoi, reverse=reverse)
        else: test_ds = None

        return cls(train_ds, val_ds, test_ds, bs)

    @classmethod
    def from_folders(cls, bs, labels, train, val=None, test=None, reverse=False, stoi=None):

        train_ds = TextDataset.from_text_files(train, labels, stoi=stoi, reverse=reverse)
        stoi = train_ds.stoi

        if val: val_ds = TextDataset.from_text_files(val, labels, stoi=stoi, reverse=reverse)
        else: train_ds, val_ds = train_ds.split_train_val()
            
        if test: test_ds = TextDataset.from_text_files(test, labels, stoi=stoi, reverse=reverse)
        else: test_ds = None

        return cls(train_ds, val_ds, test_ds, bs=bs)


# SECTION (3) - PYTORCH MODELS

class LockedDropout(nn.Module):
    "Locked dropout layer. Input is 3D tensor, dropout along dimension 0 is constant at every forward call."

    def __init__(self, drop):
        super().__init__()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        mask = self.drop(torch.ones(1, x.size(1), x.size(2)).cuda())
        return mask * x

class EmbeddingDropout(nn.Module):
    "Embedding Dropout layer for word embeddings."

    def __init__(self, vocab_size, emb_dim, drop1, drop2, pad_token):
        super().__init__()
        self.vocab_size, self.pad_token = vocab_size, pad_token
        self.drop1, self.drop2 = nn.Dropout(drop1), LockedDropout(drop2)
        self.embed = nn.Embedding(vocab_size, emb_dim, pad_token)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.constant_(self.embed.weight[pad_token], 0)

    def forward(self, x):        
        # Input Shape: seqlen x bs
        # Output Shape: seqlen x bs x emb_dim
        
        if self.training == False:
            return self.embed(x)
        
        if self.training == True:
            mask = self.drop1(torch.ones(self.vocab_size,1).cuda())
            x = F.embedding(x, self.embed.weight * mask, self.pad_token)
            return self.drop2(x)
        
class WeightDropLSTM1(nn.Module):
    "Class for a single layer weight-dropped LSTM cell, wraps pytorch LSTM class."

    def __init__(self, input_size, hidden_size, drop):
        super().__init__()
        self.weight_drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.setup_raw()
        
    def setup_raw(self):
        weight_hh_l0 = getattr(self.lstm, 'weight_hh_l0')
        self.lstm.register_parameter('weight_hh_l0_raw', weight_hh_l0)        

    def clear_non_raw(self):
        weight_hh_l0 = getattr(self.lstm, 'weight_hh_l0')
        del self.lstm._parameters['weight_hh_l0']
        setattr(self.lstm, 'weight_hh_l0', weight_hh_l0*0)
    
    def forward(self, x, h0c0):
        
        # Input: x, h0c0 = (h0,c0)
        # x shape: (seqlen x bs x input_size)
        # h0,c0 shape: (num_layers=1 x bs x hidden_size)

        # Output: y, (hn, cn)
        # y shape: (seqlen x bs x hidden_size)
        # hn,cn shape: (num_layers=1 x bs x hidden_size)
        
        # extract raw copy of weight_hh_l0 and perform weight drop on it
        weight_hh_l0_raw = getattr(self.lstm, 'weight_hh_l0_raw')
        weight_hh_l0 = self.weight_drop(weight_hh_l0_raw)
        if hasattr(self.lstm, 'weight_hh_l0'): delattr(self.lstm, 'weight_hh_l0')
        setattr(self.lstm, 'weight_hh_l0', weight_hh_l0)
        
        # pass input x through the model (with the dropped-weight weight matrix)
        self.lstm.flatten_parameters()
        return self.lstm(x, h0c0)

class LSTM_Encoder(nn.Module):
    """ Class for a multi-layer LSTM Encoder with weight-drop for LSTM cells +
        dropout used for word embeddings and hidden states of LSTM intermediate layers. 
        Used as the encoder for both language modeling and text classification. """

    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, pad_token, drops, bs):
        
        super().__init__()
        emb_drop1, emb_drop2, weight_drop, hidden_drop = drops        
        self.word_embed = EmbeddingDropout(vocab_size, emb_dim, emb_drop1, emb_drop2, pad_token)
        self.hidden_drop = LockedDropout(hidden_drop)
        self.sizes = [emb_dim] + (num_layers-1)*[hidden_size] + [emb_dim]
        self.lstms = nn.ModuleList([WeightDropLSTM1(self.sizes[i], self.sizes[i+1], weight_drop) 
                                    for i in range(num_layers)])
        self.reset(bs)
    
    def reset(self, bs):
        self.h = [torch.zeros(1, bs, self.sizes[i]).cuda() for i in range(1,len(self.sizes))]
        self.c = [torch.zeros(1, bs, self.sizes[i]).cuda() for i in range(1,len(self.sizes))]
    
    def forward(self, x):
        # Input Shape: (bs x seqlen)
        # Output Shape: (seqlen x bs x emb_dim)

        h_n, c_n = [],[]
        x = x.transpose(1,0)
        x = self.word_embed(x)
        
        for i,lstm in enumerate(self.lstms):
            h0_i, c0_i = self.h[i], self.c[i]
            x, (hn_i,cn_i) = lstm(x,(h0_i,c0_i))
            if i < len(self.lstms): x = self.hidden_drop(x)
            h_n.append(hn_i.detach_())
            c_n.append(cn_i.detach_())

        self.h, self.c = h_n, c_n
        return x

class LanguageModelDecoder(nn.Module):
    """Simple linear decoder for an input encoded by LSTM_Encoder.
       Uses dropout and weight tying with embedding layer of encoder.
       """

    def __init__(self, vocab_size, emb_dim, drop, tied_weight):
        super().__init__()
        self.lin = nn.Linear(emb_dim, vocab_size, bias=False)
        self.drop = LockedDropout(drop)
        self.lin.weight = tied_weight

    def forward(self, enc_out):
        
        # Input: The input 'enc_out' is the output of LSTM_Encoder.         
        # Output: Returns 'pred_tokens', 'enc_out'
        #         * enc_out is same as input.
        #         * pred_tokens is (bs x vocab_size x seqlen) Tensor of 
        #           predicted probs of each tok in vocab.
       
        pred_tokens = self.lin(self.drop(enc_out)).permute(1,2,0)
        return pred_tokens, enc_out 

class TextClassificationDecoder(nn.Module):
    """Text Classification decoder for input encoded by LSTM_Encoder.
       Uses attention mechanism to combine encoder outputs, and then passes resulting
       combined tensor through a user specified fully connected net of arbitrary depth. """

    def __init__(self, emb_dim, num_classes, attn_size, fc_layer_sizes, fc_drops):
        super().__init__()
        fc_layer_sizes = [emb_dim] + fc_layer_sizes + [num_classes]
        self.fc = FullyConnectedNet(fc_layer_sizes, fc_drops)
        self.attn1 = nn.Linear(emb_dim, attn_size)
        self.attn2 = nn.Linear(attn_size, 1)
        initialize_modules([self.attn1, self.attn2], nn.init.kaiming_normal_)

    def forward(self, enc_in, enc_out):
        
        # Input: 'enc_in' and 'enc_out' are the input and output of LSTM_Encoder.
        # Output: Returns 'pred_classes', 'attn'
        #         * pred_classes: Tensor of shape (bs x num_classes), giving predicted probs of classes.
        #         * attn: Tensor of attention weights of shape (seqlen x bs), each column sums to 1.
        
        
        # attention values
        attn = F.relu(self.attn1(enc_out))                       # seqlen x bs x attn_size
        attn = self.attn2(attn).squeeze()                        # seqlen x bs
        attn = F.softmax(attn, dim=0)                            # seqlen x bs
        attn = attn * (enc_in.transpose(1,0) != 1).float()       # seqlen x bs (ignore pad token)
        attn = attn / attn.sum(dim=0).unsqueeze(0)               # seqlen x bs
        
        # combined encoder output (using attention weighting)
        weighted_enc_out = attn.unsqueeze(2) * enc_out           # seqlen x bs x emb_dim
        combined_enc_out = weighted_enc_out.sum(0)               # bs x emb_dim 
        
        # pred_classes
        pred_classes = self.fc(combined_enc_out)                 # bs x num_classes
        return pred_classes, attn
        
class LanguageModelNet(nn.Module):
    """ Two part language model architecture consisting of an LSTM encoder and linear decoder.
    
       If pretrained == 'fwd' or 'bwd', weights are loaded from a model pretrained
       on the wiki_text103 corpus with texts in 'fwd' or 'bwd' direction.
       If pretrained == None, then no pretrained weights are loaded. 
       
       Original weights from the fastai implementation of the pretrained model are available at 
       "http://files.fast.ai/models/wt103/". Our implementation is signicantly restructured 
       code-wise from fastai implementation, but is almost equivalent mathematically, and uses 
       same pretrained weight matrices in initialization. 
       """

    def __init__(self, data, enc_drops=[0.05, 0.25, 0.2, 0.15], dec_drop=0.1, drop_scaling=0.7, pretrained=None):
        super().__init__()
        
        # basic parameters and itos/stoi mappings
        enc_drops, dec_drop = list_mult(enc_drops,drop_scaling), dec_drop*drop_scaling
        emb_dim, hidden_size, num_layers = 400, 1150, 3            # values for pretrained model
        vocab_size, pad_token= len(data.stoi), data.stoi['_pad_']  # values depending on data
        self.bs, self.stoi, self.itos = data.bs, data.stoi, {i:s for s,i in data.stoi.items()}
        
        # encoder
        self.enc = LSTM_Encoder(vocab_size, emb_dim, hidden_size, num_layers, pad_token, enc_drops, self.bs)             
        if pretrained: self.load_weights(pretrained)    
        
        # decoder
        self.dec = LanguageModelDecoder(vocab_size, emb_dim, dec_drop, tied_weight = self.enc.word_embed.embed.weight)
        
        # layer groups and param groups
        self.head = self.dec
        self.layer_groups = [self.enc.lstms, self.head]
        self.param_groups = separate_bn_layers(self.layer_groups)

    def reset(self):
        self.enc.reset(self.bs)
        
    def clear_non_raw(self):
        self.cuda()
        for lstm in self.enc.lstms: lstm.clear_non_raw()
    
    def forward(self,x): 
        return self.dec(self.enc(x))
    
    def predict_from_string(self,s,n,k=5):
        # Predict next n tokens following a string s, and return as a new string s2.
        # At each step, next token to add is chosen out of top k possibilities, 
        # according to their relative probabilities.
        
        self.eval()
        self.enc.reset(bs=1)
        s = tokenize([s])
        s, _ = numericalize(s, stoi=self.stoi)
        s = s[0]
        for i in range(n):
            ss = torch.LongTensor(s).unsqueeze(0).cuda()
            probs_next_token = F.softmax(self.forward(ss)[0][0,:,-1],dim=0)
            probs_next_token[:4] = 0 # special tokens
            top_probs, top_indices  = torch.topk(probs_next_token,k)
            top_probs_dist = torch.distributions.categorical.Categorical(probs=top_probs/top_probs.sum())
            next_token = top_indices[top_probs_dist.sample()].item()
            s.append(next_token)
            
        self.enc.reset(bs=self.bs)
        s = ' '.join([self.itos[x] for x in s])
        return s

    def load_weights(self, pretrained):
        rel_path = '../Applications/TextModels/'

        with open(rel_path + 'stoi_wt103.pkl', 'rb') as f:
            stoi_wt103 = pickle.load(f)

        if pretrained == 'fwd':
            self.enc.lstms.load_state_dict(torch.load(rel_path + 'fwd_lstms_wt103.pt'), strict=False)
            emb_weights_pretrained = torch.load(rel_path + 'fwd_embed_wt103.pt')['weight']
            emb_mean = emb_weights_pretrained.mean(dim=0)
            emb_weight = torch.zeros(len(self.itos), 400)
            for i,s in self.itos.items(): 
                if s in stoi_wt103: emb_weight[i] = emb_weights_pretrained[stoi_wt103[s]]
                else: emb_weight[i] = copy.deepcopy(emb_mean)
            setattr(self.enc.word_embed.embed, 'weight', nn.Parameter(emb_weight.detach_()))
            
        elif pretrained == 'bwd':
            self.enc.lstms.load_state_dict(torch.load(rel_path + 'bwd_lstms_wt103.pt'), strict=False)
            emb_weights_pretrained = torch.load(rel_path + 'bwd_embed_wt103.pt')['weight']
            emb_mean = emb_weights_pretrained.mean(dim=0)
            emb_weight = torch.zeros(len(self.itos), 400)
            for i,s in self.itos.items(): 
                if s in stoi_wt103: emb_weight[i] = emb_weights_pretrained[stoi_wt103[s]]
                else: emb_weight[i] = copy.deepcopy(emb_mean)
            setattr(self.enc.word_embed.embed, 'weight', nn.Parameter(emb_weight.detach_()))

class TextClassificationNet(nn.Module):
    """ Two part text classification architecture consisting of an LSTM encoder and 
        attention-based decoder to combine encoder outputs.
        
        A language model trained on the same dataset is passed in to initialize. 
        The LSTM Encoder for the text classifier is the same as the LSTM Encoder 
        for the language model, except possibly with different levels of dropout. 
        The weights for the text classification encoder are initialized with the 
        corresponding weights of the language model encoder. """

    def __init__(self, PATH, language_model, num_classes, attn_size=100, enc_drops=[0.05, 0.25, 0.2, 0.15],
                 drop_scaling = 0.7, fc_layer_sizes=[100], fc_drops=[0.25,0.25]):
        
        super().__init__()
        
        # basic parameters
        enc_drops = list_mult(enc_drops, drop_scaling)
        emb_dim, hidden_size, num_layers = 400, 1150, 3
        vocab_size, pad_token= len(language_model.stoi), language_model.stoi['_pad_']
        self.bs, self.stoi = language_model.bs, language_model.stoi 
        
        # save language_model encoder state_dict
        PATH = correct_foldername(PATH)
        os.makedirs(PATH + 'models', exist_ok=True)
        torch.save(language_model.enc.state_dict(), PATH + 'models/lang_model_enc.pt')
        
        # define encoder
        self.enc = LSTM_Encoder(vocab_size, emb_dim, hidden_size, num_layers, pad_token, enc_drops, self.bs)
        self.enc.load_state_dict(torch.load(PATH + 'models/lang_model_enc.pt'), strict=False)
        
        # define decoder 
        self.dec = TextClassificationDecoder(emb_dim, num_classes, attn_size, fc_layer_sizes, fc_drops)
        
        # define layer groups and param groups 
        self.head = self.dec
        self.layer_groups = [self.enc.lstms, self.enc.word_embed, self.head]
        self.param_groups = separate_bn_layers(self.layer_groups)

    def clear_non_raw(self):
        self.cuda()
        for lstm in self.enc.lstms: lstm.clear_non_raw()    
        
    def forward(self, x, attn_vals=False):        
        self.enc.reset(len(x))
        enc_in, enc_out = x, self.enc(x)
        pred_classes, attn_values = self.dec(enc_in, enc_out)
        if attn_vals == False: return pred_classes, enc_out
        if attn_vals == True: return pred_classes, enc_out, attn_values
        
        
# SECTION (4) - LOSS FUNCTIONS AND METRICS 

class RegSeqCrossEntropyLoss(object):
    """Regularized Sequence Cross Entropy Loss.     
       Used for both language modeling and text classification. 
       Regularization is applied to output sequence of the LSTM encoder. """
    
    def __init__(self, alpha=2.0, beta=1.0):
        self.alpha, self.beta = alpha, beta
        self.cross_entropy = TEN(0.)
    
    def __call__(self, outputs, target):       
        """
        Arguments:
        outputs: outputs = (preds, enc_out) where enc_out is output of LSTM encoder.  
        target: target values for preds, must have F.cross_entropy(preds,target) defined. 
        """
        
        preds, enc_out = outputs
        loss = F.cross_entropy(preds, target)
        self.cross_entropy = TEN(loss.item()) #makes a deepcopy, deepcopy not allowed here
        if self.alpha > 0: loss += self.alpha * enc_out.pow(2).mean()
        if self.beta > 0: loss += self.beta * (enc_out[1:] - enc_out[:-1]).pow(2).mean()            
        return loss

class SeqCrossEntropyLoss(object):
    """Sequence Cross Entropy Loss.
       Extracts the actual (non-regularized) cross entropy loss 
       from a RegSeqCrossEntropyLoss instance. """
    
    def __init__(self, regularized_loss):
        self.regularized_loss = regularized_loss
        
    def __call__(self, outputs, target):
        return self.regularized_loss.cross_entropy
    
        
class LanguageModelAccuracy(object):
    "Language model accuracy (ignores special tokens)."
    
    def __call__(self, outputs, target):       
        preds,_ = outputs
        preds[:,:4,:] = 0 # special tokens
        pred_labels = preds.max(dim=1)[1]
        num_correct = (pred_labels==target).sum()
        return num_correct.float()/(target.shape[0]*target.shape[1])
    
class TextClassificationAccuracy(object):
    "Text classification accuracy" 
    
    def __call__(self, outputs, target):       
        preds,_ = outputs
        pred_labels = preds.max(dim=1)[1]
        num_correct = (pred_labels==target).sum()
        return num_correct.float()/len(target)
    
import sys
import csv
import torch

class CEFREmbed(object):
    
    def __init__(self, word_emb_dim, vocab_profile_path):
        self.embed = torch.nn.Embedding(8, word_emb_dim, padding_idx=0)
        self.vocab_profile_list = self._read_vocab_profile(vocab_profile_path)
        self.CEFR2INT = {
            'A1': 1,
            'A2': 2,
            'B1': 3,
            'B2': 4,
            'C1': 5,
            'C2': 6
        }
        self.INT2CEFR = {v: k for k, v in self.CEFR2INT.items()}
        torch.nn.init.xavier_normal_(self.embed.weight)
        # 0: trash can, 1-6: A1-C2
         
    def _read_vocab_profile(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        columns = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for i, line in enumerate(reader):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                if i == 0:
                    columns = line
                    continue
                lines.append(line)
        return self._open_multiple_words(lines)
        
    def _open_multiple_words(self, lines):
        s = {}
        for i, line in enumerate(lines):
            word = line[1]
            pos  = line[2]
            cefr = int(line[3])
            s.setdefault(
                word,
                list()
            ).append([pos, cefr])
        return s

    def _get_index(self, word):
        if word in self.vocab_profile_list:
            get_infos = self.vocab_profile_list[word]
            idxs = []
            for info in get_infos:
                # idxs.append(self.CEFR2INT[info[-1].upper()])
                idxs.append(int(info[-1]))
            return idxs
        else:
            return [0]

    def get_embed(self, word):
        idxs = self._get_index(word)
        tees = [self.embed(torch.tensor(i)).unsqueeze(0) for i in idxs]
        wbed = torch.mean(torch.cat(tees, dim=0), dim=0)
        assert len(wbed.shape) == 1
        return wbed
    
    def get_cefr_tags(self, word):
        if word in self.vocab_profile_list:
            get_infos = self.vocab_profile_list[word]
            cefrs = []
            for info in get_infos:
                cefrs.append(self.INT2CEFR[int(info[-1])])
            return cefrs
        else:
            return [0]
        
    def get_CEFR2INT(self):
        return self.CEFR2INT

    def get_INT2CEFR(self):
        return self.INT2CEFR
    
    def get_CEFR2INT_w_zero(self):
        a = self.CEFR2INT
        a[0] = 0
        return a
    
    def get_INT2CEFR_w_zero(self):
        a = self.INT2CEFR
        a[0] = 0
        return a

class FILLEDEmbed(object):
    def __init__(self, word_emb_dim, filled_pauses_path):
        self.embed = torch.nn.Embedding(2, word_emb_dim, padding_idx=0)
        torch.nn.init.xavier_normal_(self.embed.weight)
        self.filled_pauses_list = self._read_filled_pauses(filled_pauses_path)
        # 0: not filled pauses, 1: hesitation, pause tokens
        
    def _read_filled_pauses(self, filled_pauses_path):
        r = []
        with open(filled_pauses_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                r.append(line)
        return r

    def _get_index(self, word):
        if word in self.filled_pauses_list:
            return [1] # filled pauses tag
        return [0]

    def get_embed(self, word):
        idxs = self._get_index(word)
        tees = [self.embed(torch.tensor(i)).unsqueeze(0) for i in idxs]
        wbed = torch.mean(torch.cat(tees, dim=0), dim=0)
        assert len(wbed.shape) == 1
        return wbed
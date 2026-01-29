import torch
from typing import List, Union


class Vocabulary:
    """
    Unified vocabulary interface.
    Handles token <-> ID conversion and special tokens.
    """
    
    def __init__(
        self,
        tokens: List[str],
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ):
        """
        Args:
            tokens: List of vocabulary tokens (can include special tokens)
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            pad_token: Padding token
            unk_token: Unknown token
        """
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        self.token2id = {}
        self.id2token = []
        
        # Add special tokens first
        for token in [bos_token, pad_token, eos_token, unk_token]:
            if token not in self.token2id:
                idx = len(self.id2token)
                self.token2id[token] = idx
                self.id2token.append(token)
        
        self.nspecial = len(self.id2token)
        
        # Add vocabulary tokens
        for token in tokens:
            if token not in self.token2id:
                idx = len(self.id2token)
                self.token2id[token] = idx
                self.id2token.append(token)
    
    @classmethod
    def from_fairseq(cls, dict_path: str):
        """
        Load from fairseq-style dictionary.
        
        Format:
            <s> 0
            <pad> 0
            </s> 0
            <unk> 0
            AH 35554
            N 25395
            ...
        """
        tokens = []
        special_tokens = {}
        
        with open(dict_path, 'r') as f:
            for line in f:
                token = line.split()[0]
                # Detect special tokens
                if token in ['<s>', '<pad>', '</s>', '<unk>', '<SIL>', 'sil']:
                    if token == '<s>':
                        special_tokens['bos_token'] = token
                    elif token in ['<pad>']:
                        special_tokens['pad_token'] = token
                    elif token == '</s>':
                        special_tokens['eos_token'] = token
                    elif token == '<unk>':
                        special_tokens['unk_token'] = token
                tokens.append(token)
        
        return cls(tokens, **special_tokens)
    
    @classmethod
    def from_text(cls, text_path: str, min_freq: int = 1):
        """
        Build vocabulary from text file by counting tokens.
        
        Args:
            text_path: Path to text file (one sequence per line, space-separated)
            min_freq: Minimum frequency to include token
        """
        from collections import Counter
        
        counter = Counter()
        with open(text_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                counter.update(tokens)
        
        # Keep tokens above min_freq
        tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
        
        return cls(tokens)
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> torch.LongTensor:
        """
        Encode text to token IDs.
        
        Args:
            text: Space-separated token string
            add_bos: Prepend BOS token
            add_eos: Append EOS token
        
        Returns:
            Tensor of token IDs
        """
        tokens = text.strip().split()
        ids = []
        
        if add_bos:
            ids.append(self.bos_id)
        
        for token in tokens:
            ids.append(self.token2id.get(token, self.unk_id))
        
        if add_eos:
            ids.append(self.eos_id)
        
        return torch.LongTensor(ids)
    
    def decode(self, ids: Union[torch.Tensor, List[int]], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: Tensor or list of token IDs
            skip_special: Skip special tokens in output
        
        Returns:
            Space-separated token string
        """
        if torch.is_tensor(ids):
            ids = ids.tolist()
        
        tokens = []
        special_ids = {self.bos_id, self.pad_id, self.eos_id, self.unk_id} if skip_special else set()
        
        for idx in ids:
            if idx not in special_ids and idx < len(self.id2token):
                tokens.append(self.id2token[idx])
        
        return ' '.join(tokens)
    
    @property
    def bos_id(self) -> int:
        return self.token2id[self.bos_token]
    
    @property
    def eos_id(self) -> int:
        return self.token2id[self.eos_token]
    
    @property
    def pad_id(self) -> int:
        return self.token2id[self.pad_token]
    
    @property
    def unk_id(self) -> int:
        return self.token2id[self.unk_token]
    
    def __len__(self) -> int:
        return len(self.id2token)
    
    def save(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            for token in self.id2token:
                f.write(f"{token}\n")
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            tokens = [line.strip() for line in f]
        return cls(tokens)
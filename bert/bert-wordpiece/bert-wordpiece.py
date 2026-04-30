from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        if len(word) > self.max_word_len:
            return [self.unk_token]
            
        first = True
        ret = []
        while word: # FIX 1: removed 'not'
            found_substring = False
            # We try prefixes of 'word' from longest to shortest
            for i in range(len(word), 0, -1):
                prefix = word[:i]
                token = prefix if first else "##" + prefix
                
                if token in self.vocab:
                    ret.append(token)
                    word = word[i:] # Keep the remainder
                    first = False   # FIX 2: update the flag
                    found_substring = True
                    break
            
            # FIX 3: If no prefix matches, the whole word is unknown
            if not found_substring:
                return [self.unk_token]
                
        return ret

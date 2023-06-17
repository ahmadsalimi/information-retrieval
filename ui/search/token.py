from dataclasses import dataclass


@dataclass
class Token:
    processed: str
    actual: str
    i: int
    idx: int

    @staticmethod
    def from_spacy_token(token) -> "Token":
        return Token(token.lemma_.lower(), token.text, token.i, token.idx)

    def __repr__(self):
        return self.processed

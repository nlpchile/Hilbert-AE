class Tokens:
    def __init__(self):

        self.token2index = {}
        self.index2token = []

    def add_token(self, token):

        if token not in self.token2index:
            self.index2token.append(token)
            self.token2index[token] = len(self.index2token) - 1

        return self.token2index[token]

    def __len__(self) -> int:
        return len(self.index2token)
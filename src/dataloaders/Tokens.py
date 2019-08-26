from typing import Any


class Tokens:
    def __init__(self) -> None:

        self.token2index = {}
        self.index2token = []

    def add_token(self, token: Any) -> int:
        '''

        This method updates the class dictionaries adding
        a new token to its vocabulary.

        Args:

            token (Any) : The token to be added.

        Returns:

            (int) : It returns the index for the given token.

        '''

        if token not in self.token2index:
            self.index2token.append(token)
            self.token2index[token] = len(self.index2token) - 1

        return self.token2index[token]

    def __len__(self) -> int:
        return len(self.index2token)
"""This module implements the Token Classes."""

from typing import Any, Dict, List


class Tokens:
    """Tokens Class."""

    def __init__(self) -> None:
        """Tokens Class."""
        self.token2index: Dict[Any, int] = {}
        self.index2token: List[Any] = []

    def add_token(self, token: Any) -> int:
        """

        Update the internal class state adding a new token to its vocabulary and index.

        Args:
            token (Any) : The token to be added.

        Returns:
            (int) : It returns the index for the given token.

        """
        if token not in self.token2index:
            self.index2token.append(token)
            self.token2index[token] = len(self.index2token) - 1

        return self.token2index[token]

    def __len__(self) -> int:
        """Return the length of the vocabulary."""
        return len(self.index2token)

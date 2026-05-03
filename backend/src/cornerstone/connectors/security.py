from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


class TokenCipher:
    """Small encryption boundary for connector credentials.

    The application stores only ciphertext. Provider adapters receive plaintext only at call time.
    This intentionally keeps token handling centralized before moving to a managed KMS.
    """

    def __init__(self, secret: str) -> None:
        if not secret or len(secret.strip()) < 16:
            raise ValueError("connector_encryption_secret must be at least 16 characters")
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(digest))

    def encrypt(self, value: str) -> str:
        if value == "":
            raise ValueError("Cannot encrypt empty connector secret")
        return self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Connector credential could not be decrypted") from exc

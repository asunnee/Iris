from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7
from cryptography.hazmat.backends import default_backend
import os


def encrypt(key, plaintext):

    key = bytes.fromhex(key)

    iv = os.urandom(16)

    padder = PKCS7(128).padder()

    padded = padder.update(plaintext) + padder.finalize()

    cipher = Cipher(
        algorithms.AES(key[:32]),
        modes.CBC(iv),
        backend=default_backend()
    )

    encryptor = cipher.encryptor()

    ciphertext = encryptor.update(padded) + encryptor.finalize()

    return iv, ciphertext


def decrypt(key, iv, ciphertext):

    key = bytes.fromhex(key)

    cipher = Cipher(
        algorithms.AES(key[:32]),
        modes.CBC(iv),
        backend=default_backend()
    )

    decryptor = cipher.decryptor()

    padded = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = PKCS7(128).unpadder()

    plaintext = unpadder.update(padded) + unpadder.finalize()

    return plaintext
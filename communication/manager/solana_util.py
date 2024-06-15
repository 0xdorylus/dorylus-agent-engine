from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.signature import Signature
from typing import List, Tuple


def generate_keypair():
    return Keypair()

def restore_keypair(s: str):
    return Keypair.from_base58_string(s)

def sign_message(kp: Keypair, message: str) -> Tuple[str, str]:
    message_bytes = message.encode('utf-8')
    signature = kp.sign_message(message_bytes)
    return bytes(signature).hex(), message_bytes.hex()

def verify_signature(pubkey: Pubkey, signature_hex: str, message_hex: str) -> bool:
    signature = Signature.from_bytes(bytes.fromhex(signature_hex))
    return signature.verify(pubkey, bytes.fromhex(message_hex))

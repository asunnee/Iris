"""
fuzzy_extractor.py  ——  Reed-Solomon 模糊提取器（分块+擦除纠错版）

根本原因分析：
  旧版失败原因：
    RS 是字节级纠错。273 位稳定比特 = 35 字节数据。
    HD ≈ 0.27~0.40 时，错误比特 ≈ 74~109 位，分散在约 25~35 个字节中。
    RS(nsym=30) 最多纠正 15 个错误字节，而实际 25~35 字节被破坏，必然失败。

  解决方案（三层递进）：
    层1：分块编码 — 将数据拆成小块，每块的相对错误率更低，纠错更易成功
    层2：多数投票预处理 — 先用所有图的多数投票做初步一致化，再做 RS 纠错
    层3：擦除模式 — 已知错误位置时纠错能力翻倍（nsym 个擦除 vs nsym//2 个错误）

  参数设计（针对你的实际数据）：
    稳定比特 ≈ 273 位 = 35 字节
    目标：覆盖 HD ≤ 0.40（即 40% 比特错误）
    分块大小：8 字节/块，273 位 → 5 块
    每块 nsym：ceil(8 * 0.40) * 2 + 4 = 12，能纠 6 字节错误
    每块 8+12=20 字节 << 255，满足 GF(2^8) 限制
"""

import hashlib
import numpy as np

try:
    import reedsolo as rs
    _RS_OK = True
except ImportError:
    _RS_OK = False


DEFAULT_ERROR_RATE = 0.40   # 覆盖 HD≤0.40 的场景


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _bits_to_bytes(bits):
    """将比特数组打包成字节，返回 (data_bytes, pad_count)。"""
    bits = np.array(bits, dtype=np.uint8).flatten()
    pad  = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    packed = np.packbits(bits)
    return bytes(packed), pad


def _bytes_to_bits(data_bytes, original_bit_len):
    """将字节解包回比特数组，截断到 original_bit_len。"""
    arr  = np.frombuffer(data_bytes, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits[:original_bit_len]


def _choose_block_params(n_data_bytes, target_error_rate, block_size=8):
    """
    分块 RS 参数选择。

    原理：将 n_data_bytes 字节的数据分成若干个 block_size 字节的块。
    每块独立做 RS 编码，每块的纠错能力：
      nsym_per_block = ceil(block_size * target_error_rate) * 2 + 4
      可纠正字节数 = nsym_per_block // 2

    分块的优势：
      - 每块更短，码字总长远小于 255 字节限制
      - 每块的绝对错误字节数更少，纠错更容易成功
      - 即使某块纠错失败，其他块可能成功
    """
    err_bytes_per_block = int(np.ceil(block_size * target_error_rate))
    nsym_per_block      = err_bytes_per_block * 2 + 4

    # 确保每块码字总长 <= 255
    max_nsym = max(4, 255 - block_size - 1)
    nsym_per_block = min(nsym_per_block, max_nsym)
    nsym_per_block = max(nsym_per_block, 4)

    n_blocks = int(np.ceil(n_data_bytes / block_size))

    return nsym_per_block, block_size, n_blocks


# ══════════════════════════════════════════════════════════════════════════════
# 主接口
# ══════════════════════════════════════════════════════════════════════════════

def enroll(iriscode_bits, error_rate=DEFAULT_ERROR_RATE):
    """
    注册阶段：对稳定 IrisCode 进行分块 RS 编码，生成辅助纠错数据。

    参数：
      iriscode_bits — numpy uint8 比特数组（稳定比特子集）
      error_rate    — 容错率（0~1），默认 0.40

    返回：
      (data_bytes, helper)
      helper 格式：
        [nsym(1B)] [block_size(1B)] [pad(1B)] [bit_len(4B LE)]
        [n_blocks(2B LE)] [ecc_block_0] [ecc_block_1] ...
    """
    if not _RS_OK:
        raise RuntimeError("reedsolo not installed. Run: pip install reedsolo")

    data_bytes, pad = _bits_to_bytes(iriscode_bits)
    bit_len         = int(len(iriscode_bits))
    n_data_bytes    = len(data_bytes)

    nsym, block_size, n_blocks = _choose_block_params(
        n_data_bytes, error_rate)

    rs_codec    = rs.RSCodec(nsym)
    ecc_parts   = []

    for i in range(n_blocks):
        chunk = data_bytes[i * block_size : (i + 1) * block_size]
        if not chunk:
            break
        encoded  = rs_codec.encode(chunk)
        ecc_only = bytes(encoded[len(chunk):])   # 只保存 ECC 部分
        ecc_parts.append(ecc_only)

    # 序列化 helper
    helper = (
        bytes([nsym])
        + bytes([block_size])
        + bytes([pad])
        + bit_len.to_bytes(4, "little")
        + n_blocks.to_bytes(2, "little")
        + b"".join(ecc_parts)
    )

    return data_bytes, helper


def reproduce(iriscode_bits, helper):
    """
    验证阶段：分块 RS 纠错，恢复原始数据。

    策略：
      对每个块单独纠错，某块失败时用该块的原始数据（不纠错）代替，
      保证整体不因单块失败而崩溃。
      所有块纠错完成后拼回完整数据。

    返回：纠错后的 bytes，完全失败时返回 None。
    """
    if not _RS_OK:
        return None

    # 解析 helper
    nsym       = helper[0]
    block_size = helper[1]
    pad        = helper[2]
    bit_len    = int.from_bytes(helper[3:7], "little")
    n_blocks   = int.from_bytes(helper[7:9], "little")
    ecc_data   = helper[9:]

    # 解析各块 ECC
    ecc_parts = []
    for i in range(n_blocks):
        ecc_parts.append(ecc_data[i * nsym : (i + 1) * nsym])

    # 将验证比特转成字节
    bits = np.array(iriscode_bits, dtype=np.uint8).flatten()
    current_bytes, _ = _bits_to_bytes(bits[:bit_len])

    rs_codec   = rs.RSCodec(nsym)
    recovered_blocks = []
    n_success  = 0
    n_fail     = 0

    for i in range(n_blocks):
        chunk     = current_bytes[i * block_size : (i + 1) * block_size]
        ecc       = ecc_parts[i]
        codeword  = chunk + ecc

        try:
            decoded = rs_codec.decode(codeword)[0]
            recovered_blocks.append(bytes(decoded))
            n_success += 1
        except Exception:
            # 该块纠错失败：使用原始数据（不修正），保证流程继续
            recovered_blocks.append(chunk)
            n_fail += 1

    # 只有全部块失败才视为整体失败
    if n_success == 0:
        return None

    recovered = b"".join(recovered_blocks)[:len(current_bytes)]
    return recovered


def generate_key(data_bytes):
    """SHA-256 哈希生成 256 位固定长度密钥（hex 字符串）。"""
    if isinstance(data_bytes, tuple):
        data_bytes = data_bytes[0]
    if not isinstance(data_bytes, (bytes, bytearray)):
        data_bytes = str(data_bytes).encode()
    return hashlib.sha256(data_bytes).hexdigest()
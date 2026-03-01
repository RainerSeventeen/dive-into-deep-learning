"""字节对编码

    算法复杂度偏高, 展示原理和步骤
"""
from collections import defaultdict

def get_max_freq_pair(token_freqs):
    """找到当前频率最高的那一对"""
    pairs = defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split() # 切分成单独的字符
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get) # 返回最大的 pairs 键

def merge_symbols(max_freq_pair, token_freqs, symbols):
    """将最大的那一对, 去掉空格视为一个 token 并重新统计"""
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = {}
    for token, _ in token_freqs.items():
        # 把 token 中原来的空格压缩掉, 视为一个 token
        new_token = token.replace(' '.join(max_freq_pair), ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs


if __name__ == "__main__":
    symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '_', '[UNK]']

    # 因为不考虑跨单词, 所以在结尾处增加一个标记表示单词终止
    raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
    token_freqs = {}

    for token, freq in raw_token_freqs.items():
        token_freqs[' '.join(list(token))] = raw_token_freqs[token]

    print(f"1. {token_freqs}")

    num_merges = 10
    for i in range(num_merges):
        max_freq_pair = get_max_freq_pair(token_freqs)
        token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
        print(f'合并# {i+1}:',max_freq_pair)
    
    print(symbols)
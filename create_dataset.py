import hashlib
import re
import urllib.request
from collections import Counter
from pathlib import Path
import numpy as np

URL = 'https://www.gutenberg.org/cache/epub/11/pg11.txt'

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower().replace('’', "'"))

def prepare_data(source=None, max_words=20000, context=20, max_vocab=3000):
    if source:
        raw_bytes = Path(source).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=45) as response:
            raw_bytes = response.read()
    raw = raw_bytes.decode('utf-8-sig')
    # Remove front matter/contents and Gutenberg footer before tokenizing.
    start = re.search(r'Alice was beginning', raw, re.I)
    end = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG', raw, re.I)
    if not start or not end:
        raise ValueError('Unexpected source format: story boundaries not found.')
    words = tokenize(raw[start.start():end.start()])[:max_words]
    n = len(words)
    parts = {'train': words[:int(.8*n)], 'val': words[int(.8*n):int(.9*n)],
             'test': words[int(.9*n):]}
    if min(map(len, parts.values())) <= context:
        raise ValueError('Increase max_words or reduce context.')
    vocab = ['<PAD>', '<UNK>'] + [w for w, _ in Counter(parts['train']).most_common(max_vocab-2)]
    lookup = {w: i for i, w in enumerate(vocab)}
    arrays, stats = {}, {}
    for split, tokens in parts.items():
        ids = np.array([lookup.get(w, 1) for w in tokens], dtype=np.int32)
        # Each target uses only words within its own split. No overlapping split windows.
        windows = np.lib.stride_tricks.sliding_window_view(ids, context+1).copy()
        arrays[split] = (windows[:, :-1].copy(), windows[:, -1].copy())
        stats[split] = {'words': len(ids), 'examples': len(windows),
                        'oov_rate': float(np.mean(ids == 1))}
    return arrays, vocab, {'source': URL, 'raw_sha256': hashlib.sha256(raw_bytes).hexdigest(),
                           'words': n, 'vocab_size': len(vocab), 'context': context, 'splits': stats}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Create the shared next-word dataset once.')
    parser.add_argument('--source', help='Optional local copy of the original Gutenberg text')
    args = parser.parse_args()
    arrays, vocabulary, metadata = prepare_data(source=args.source)
    target = Path(__file__).resolve().parent / 'dataset.npz'
    values = {f'{split}_{part}': arrays[split][index]
              for split in arrays for index, part in enumerate(['x', 'y'])}
    values['vocabulary'] = np.asarray(vocabulary)
    values['metadata_json'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(target, **values)
    print(json.dumps(metadata, indent=2))
    print('Saved:', target)

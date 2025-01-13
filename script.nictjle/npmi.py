import os
import json
import nltk
import math
import random
import pickle
import argparse

from collections import Counter, defaultdict

def pickleStore(savethings , filename):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return

def combine_into_documents(data_path):
    documents = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            documents.setdefault(
                e.get('speaker'),
                []
            ).extend([e.get('text')])
    return documents

def iter_ngrams(words, n):
    """Iterate over all word n-grams in a list."""
    if len(words) < n:
        yield words

    for i in range(len(words) - n + 1):
        yield words[i:i+n]

def trans_lowers_text_list(text):
    return [t.lower() for t in text.split()]

def calculate_pmi(vocab, documents, window_width):
    doc_pmi = {}
    for speaker, doc_text in documents.items():
        word_counts = Counter()
        cooccur_counts = defaultdict(Counter)
        pmi = defaultdict(Counter)
        max_pmi = 0.
        for ngram in iter_ngrams(trans_lowers_text_list(doc_text), n=window_width):
            for i, src_word in enumerate(ngram):
                if src_word not in vocab:    # OOV
                    continue
                word_counts[src_word] += 1
                for j, tgt_word in enumerate(ngram):
                    if i != j and tgt_word != 0:
                        cooccur_counts[src_word][tgt_word] += 1

        log_total_counts = math.log(sum(word_counts.values()))
        for src_word, tgt_word_counts in cooccur_counts.items():
            for tgt_word, counts in tgt_word_counts.items():
                unconstrained_pmi = log_total_counts + math.log(counts)
                unconstrained_pmi -= math.log(word_counts[src_word] *
                                                word_counts[tgt_word])
                if unconstrained_pmi > 0.:
                    pmi[src_word][tgt_word] = unconstrained_pmi
                    max_pmi = max(max_pmi, unconstrained_pmi)
        doc_pmi[speaker] = {'pmi': pmi, 'max_pmi': max_pmi}
    return doc_pmi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')
    parser.add_argument('--pmi_window_width', type=int, default=2, help='Window size for calculating PMI, which is disabled when -1')

    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    vocab = {word: word_id for word, word_id in keys}
    documents = {speaker: ' '.join(list(sum(paragraphs, []))) for speaker, paragraphs in combine_into_documents(args.data_path).items()}
    pmi_info = calculate_pmi(vocab, documents, window_width=args.pmi_window_width)
    saveNPMI = os.path.join(save_dir, "pmi_info")
    pickleStore(pmi_info, saveNPMI)
    print("Save PMI of dataset to %s" % (saveNPMI))

import os
import argparse
import json

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def GetType(path):
    filename = path.split("/")[-1]
    return filename.split(".")[0]

def get_tfidf_embedding(text):
    """

    :param text: list, doc_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]
    """
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_weight


def compress_array(a, id2word):
    d = {}
    for i in range(len(a)):
        d[i] = {}
        for j in range(len(a[i])):
            if a[i][j] != 0:
                d[i][id2word[j]] = a[i][j]
    return d


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')
    parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='dataset name')

    args = parser.parse_args()
    sentaspara = args.sentaspara

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fname = GetType(args.data_path) + ".{}.w2d.tfidf.jsonl".format(sentaspara)
    saveFile = os.path.join(save_dir, fname)
    print("Save word2sent features of dataset %s to %s" % (args.dataset, saveFile))

    if sentaspara == 'sent':
        documents_lines = combine_into_documents(args.data_path)
        fout = open(saveFile, "w")
        for speaker, e in documents_lines.items():
            if isinstance(e, list) and isinstance(e[0], list):
                docs = list(sum(e, []))
            else:
                docs = e
            cntvector, tfidf_weight = get_tfidf_embedding(docs)
            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():
                id2word[tfidf_id] = w
            tfidfvector = compress_array(tfidf_weight, id2word)
            fout.write(json.dumps(tfidfvector) + "\n")
            
    elif sentaspara == 'para':

        fout = open(saveFile, "w")
        with open(args.data_path) as f:
            for line in f:
                e = json.loads(line)
                if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                    docs = [" ".join(doc) for doc in e["text"]]
                else:
                    docs = e["text"]
                cntvector, tfidf_weight = get_tfidf_embedding(docs)
                id2word = {}
                for w, tfidf_id in cntvector.vocabulary_.items():
                    id2word[tfidf_id] = w
                tfidfvector = compress_array(tfidf_weight, id2word)
                fout.write(json.dumps(tfidfvector) + "\n")


if __name__ == '__main__':
    main()


import re
import pickle
import sklearn.datasets as skd
from sys import argv
from collections import defaultdict
from itertools import chain

MAX_SENT_LEN = 126
LOC_GZ_OFF = 1
MISC_GZ_OFF = 2
ORG_GZ_OFF = 3
PER_GZ_OFF = 4
LOC_GZ_ON = 5
MISC_GZ_ON = 6
ORG_GZ_ON = 7
PER_GZ_ON = 8
GZ_PADDING = 9

UNKNOWN_COMPRENO_ID = 2

def get_gazetteer(filename):
    gazetteer = set()
    with open(filename, 'r') as f:
        for line in f.readlines():
            gazetteer.add(line.strip())
    return gazetteer

def max_ngram(gazetteer):
    return max(set(gazetteer), key=lambda x: len(x.split()))

def add_value_to_gazetteer_features(gazetteer_features, value):
    gazetteer_features['LOC'].insert(0, value)
    gazetteer_features['MISC'].insert(0, value)
    gazetteer_features['ORG'].insert(0, value)
    gazetteer_features['PER'].insert(0, value)
    gazetteer_features['LOC'].append(value)
    gazetteer_features['MISC'].append(value)
    gazetteer_features['ORG'].append(value)
    gazetteer_features['PER'].append(value)

# For now only sentence mode supports gazetteer features
def get_gazetteer_features(sent, gazetteers, max_n=13):
    gazetteer_features = { 'LOC': [LOC_GZ_OFF]*len(sent),
                           'MISC':[MISC_GZ_OFF]*len(sent),
                           'ORG': [ORG_GZ_OFF]*len(sent),
                           'PER': [PER_GZ_OFF]*len(sent) }
    if len(sent) < max_n:
        max_n = len(sent)

    for n in range(0, max_n):
        for i in range(len(sent)):
            word_sent = [s[0] for s in sent]
            ngram = ' '.join(word_sent[i:i+n])
            normalized_ngram = normalize(ngram)

            for key, gazetteer in gazetteers.items():
                if normalized_ngram in gazetteer:
                    for j in range(i, i + n):
                        if j >= len(sent):
                            break
                        gazetteer_features[key][j] = eval(key + '_GZ_ON')
    add_value_to_gazetteer_features(gazetteer_features, GZ_PADDING)
    flat_features = flatten(gazetteer_features.values())

    return flat_features

def flatten(array):
    return list(chain.from_iterable(array))

def init_gazetteers(gazetteers_files):
    gazetteers = {}
    for key, gzf in gazetteers_files.items():
        gazetteer = get_gazetteer(gzf)
        print(key, ' max ngram ', gzf, len(max_ngram(gazetteer).split()))
        gazetteers[key] = gazetteer

    return gazetteers

def get_senna_vecs(filename):
    senna_vecs = {}
    for parts in read_parts(filename):
        token = parts[0]
        vec = [float(x) for x in parts[1:]]
        senna_vecs[token] = vec
    return senna_vecs

def get_normalized_conll_tokens(data_dir, filenames):
    tokens = set()
    for filename in filenames:
        for parts in read_parts(data_dir + filename):
            if parts[0] not in ['-DOCSTART-', '']:
                token = normalize(parts[0])
                tokens.add(token)
    tokens.add("UNKNOWN")
    tokens.add("PADDING")

    return tokens

def get_intersect_vecs(senna_vecs, normalized_conll_tokens):
    intersect_vecs = {}
    for key, value in senna_vecs.items():
        if key in normalized_conll_tokens:
            intersect_vecs[key] = senna_vecs[key]

    return intersect_vecs

def get_label_int_map(filename):
    label_int_map = {}
    cntr = 1
    for parts in read_parts(filename):
        if parts[0] not in ['-DOCSTART-', '']:
            label = parts[-1]
            if label not in label_int_map:
                label_int_map[label] = cntr
                cntr += 1
    return label_int_map

def get_vocabulary_int_map(vocabulary):
    voc_int_map = {}
    cntr = 1
    for word in vocabulary:
        if word not in voc_int_map:
            voc_int_map[word] = cntr
            cntr += 1
    return voc_int_map

def docs_with_sents(filename):
    docs = []
    doc = []
    sent = []
    for parts in read_parts(filename):
        if parts[0] == '-DOCSTART-' and doc != []:
            docs.append(doc)
            doc = []
        elif parts[0] == '' and sent != []:
            doc.append(sent)
            sent = []
        elif parts[0] not in ['-DOCSTART-', '']:
            sent.append(parts)
    if sent != []:
        doc.append(sent)
    if doc != []:
        docs.append(doc)
    return docs

def sent_conll_in_id(docs, label_int_map, voc_int_map, gazetteers):
    save_docs = []
    line_pointer = 0
    for doc in docs:
        line_pointer += 2
        save_doc = []
        for sent in doc:
            line_pointer += 1
            save_sent = []
            sent_line_pointer = line_pointer
            for i in range(len(sent)):
                sent_word_ids = [voc_int_map['PADDING']]
                sent_cap_ids = [get_cap_id('PADDING')]
                pos_id = -(i + 1)
                sent_pos_ids = [pos_id + MAX_SENT_LEN]

                label_id = label_int_map[sent[i][-1]]
                row = [label_id]
                gazetteer_ids = get_gazetteer_features(sent, gazetteers)

                for ind, parts in enumerate(sent):
                    word = parts[0]
                    cap_id = get_cap_id(parts[0])
                    normalized = normalize(word)
                    if normalized in voc_int_map:
                        word_id = voc_int_map[normalized]
                    else:
                        word_id = voc_int_map['UNKNOWN']
                    pos_id = pos_id + 1
                    sent_pos_ids.append(pos_id + MAX_SENT_LEN)
                    sent_cap_ids.append(cap_id)
                    sent_word_ids.append(word_id)

                pos_id = pos_id + 1
                sent_pos_ids.append(pos_id + MAX_SENT_LEN)
                sent_word_ids.append(voc_int_map['PADDING'])
                sent_cap_ids.append(get_cap_id('PADDING'))

                row += sent_word_ids
                row += sent_cap_ids
                row += sent_pos_ids
                row += gazetteer_ids
                save_sent.append(row)

                line_pointer += 1
            save_doc.append(save_sent)
        save_docs.append(save_doc)
    return save_docs

def windowed_conll_in_id(docs, window, label_int_map, voc_int_map, compreno_line_to_id_map):
    windowed_docs = []
    line_pointer = 0
    for doc in docs:
        line_pointer += 2
        windowed_doc = []
        for sent in doc:
            line_pointer += 1
            windowed_sent = []
            for ind, parts in enumerate(sent):
                label_id = label_int_map[parts[-1]]
                row = [label_id]
                windowed_word_ids = []
                windowed_cap_ids = []
                compreno_ids = []
                for win_ind in range(ind - window, ind + window + 1):
                    win_compreno_line_pointer = line_pointer + win_ind - ind - 1

                    if win_ind < 0 or win_ind >= len(sent):
                        cap_id = get_cap_id('PADDING')
                        padding_id = voc_int_map['PADDING']
                        windowed_word_ids.append(padding_id)
                        windowed_cap_ids.append(cap_id)
                        compreno_ids.append(UNKNOWN_COMPRENO_ID)
                    else:
                        cap_id = get_cap_id(sent[win_ind][0])
                        word = normalize(sent[win_ind][0])
                        compreno_id = compreno_line_to_id_map[win_compreno_line_pointer]
                        if word in voc_int_map:
                            word_id = voc_int_map[word]
                        else:
                            word_id = voc_int_map['UNKNOWN']
                        windowed_word_ids.append(word_id)
                        windowed_cap_ids.append(cap_id)
                        compreno_ids.append(compreno_id)
                row += windowed_word_ids
                row += windowed_cap_ids
                row += compreno_ids
                windowed_sent.append(row)

                line_pointer += 1
            windowed_doc.append(windowed_sent)
        windowed_docs.append(windowed_doc)
    return windowed_docs

def max_sent_len(docs):
    max_len = 0
    max_sent = ''
    for doc in docs:
        for sent in doc:
            if len(sent) > max_len:
                max_len = len(sent)
                max_sent = sent
    return max_len, max_sent

def convert_to_torch_format(windowed_docs):
    in_torch_format = []
    orig_cntr = 0
    torch_cntr = 1
    orig_pos = {}
    for doc in windowed_docs:
        orig_cntr += 2
        for sent in doc:
            # Because we start counting from 1, not 0
            orig_cntr += 1
            for win in sent:
                in_torch_format.append(win)
                orig_pos[orig_cntr] = torch_cntr
                orig_cntr += 1
                torch_cntr += 1

    return in_torch_format, orig_pos

# ComprenoVec => Id
def make_compreno_id_map(id_map_filename):
    X, y = skd.load_svmlight_file(id_map_filename, n_features=83952, zero_based=True)
    id_map = {}
    for ind, row in enumerate(X):
        indexes = tuple(row.nonzero()[1].tolist())
        id_map[indexes] = int(y[ind])
    return id_map

# LinePos => Id
def make_compreno_line_to_id_map(compreno_line_map, compreno_id_map):
    compreno_line_to_id_map = {}
    for pos, compreno_vec in compreno_line_map.items():
        compreno_line_to_id_map[pos] = compreno_id_map[tuple(compreno_vec)]
    return compreno_line_to_id_map

# LinePos => ComprenoVec
def load_compreno_features_mapping(filename):
    mapping_filename = None
    if 'eng.testa.dev.iobes.subset' in filename:
        mapping_filename = 'data/conll-abbyy/conll/compreno-dev-mapping-subset.pickle'
    elif 'eng.testb.test.iobes.subset' in filename:
        mapping_filename = 'data/conll-abbyy/conll/compreno-test-mapping-subset.pickle'
    elif 'eng.train.iobes.subset' in filename:
        mapping_filename = 'data/conll-abbyy/conll/compreno-train-mapping-subset.pickle'
    elif 'eng.testa.dev.iobes' in filename:
        mapping_filename = 'data/conll-abbyy/conll/compreno-dev-mapping.pickle'
    elif 'eng.testb.test.iobes' in filename:
        mapping_filename = 'data/conll-abbyy/conll/compreno-test-mapping.pickle'
    elif 'eng.train.iobes' in filename:
        mapping_filename = 'data/conll-abbyy/conll/compreno-train-mapping.pickle'
    print(mapping_filename)
    with open(mapping_filename, 'rb') as f:
        return pickle.load(f)

# [comp_features_line_pos1, comp_features_line_pos2, ...]
def get_compreno_features_for_line(positions, compreno_mapping, fill_missed_compreno = False):
    compreno_features = []
    unknown_compreno_feature_vector = [ 83951 ]
    for orig_pos_ in sorted(positions.keys()):
        key = orig_pos_ - 1
        if fill_missed_compreno and key not in compreno_mapping:
            compreno_features.append(unknown_compreno_feature_vector)
        else:
            compreno_features.append(compreno_mapping[key])
    return compreno_features

def calc_sent_lens(docs):
    hist = defaultdict(int)
    for doc in docs:
        for sent in doc:
            hist[len(sent)] += 1
    return hist

def get_cap_id(token):
    # AllCaps, HasCap, InitCap, NoCap, Padding
    if token == 'PADDING':
        return 5
    elif token.isupper():
        return 1
    elif token[0].isupper():
        return 3
    elif any(x.isupper() for x in token):
        return 2
    else:
        return 4

def normalize(raw_token):
    token = raw_token.lower()
    token = re.sub('\d+', '0', token)
    return token

def read_parts(filename):
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            yield parts

def get_features_int_map(feature_vecs, voc_int_map):
    features_int_map = {}
    for token, vec in feature_vecs.items():
        features_int_map[voc_int_map[token]] = vec
    return features_int_map

def save_index_by_table(table, filename):
    with open(filename, 'w') as f:
        for k, v in table.items():
            if v.__class__.__name__ == 'list':
                str_v = ' '.join([str(i) for i in v])
            else:
                str_v = '{0}'.format(v)
            row = "{0}\t{1}\n".format(k, str_v)
            f.write(row)

def save_index_by_array(arrays, filename):
    with open(filename, 'w') as f:
        for array in arrays:
            label = array[0]
            vec = array[1:]
            str_vec = ' '.join([str(v) for v in vec])
            row = "{0}\t{1}\n".format(label, str_vec)
            f.write(row)

def save_index_by_array_without_label(arrays, filename):
    with open(filename, 'w') as f:
        for array in arrays:
            vec = array
            str_vec = ' '.join([str(v) for v in vec])
            row = "{0}\n".format(str_vec)
            f.write(row)

def get_filenames(mode):
    if mode == 'compreno-subset' or mode == 'compreno-subset-win':
        return ['eng.testa.dev.iobes.subset', 'eng.train.iobes.subset', 'eng.testb.test.iobes.subset']
        # return ['eng.testa.dev.iobes.subset', 'eng.train.iobes.subset']
        # return ['eng.testa.dev.iobes.subset']
    elif mode == 'sentence-convolution':
        return ['eng.testa.dev.iobes', 'eng.train.iobes', 'eng.testb.test.iobes']
    else:
        raise Exception('Incorrect mode {0}'.format(mode))


def get_filenames_for_indexing():
    return ['eng.testa.dev.iobes', 'eng.testb.test.iobes', 'eng.train.iobes']

def run_win(mode):
    w2v_file = 'data/embeddings/senna.w2v'
    data_dir = 'data/conll2003/'
    filenames = get_filenames(mode)
    filenames_for_indexing = get_filenames_for_indexing()

    senna_vecs = get_senna_vecs(w2v_file)
    normalized_conll_tokens = get_normalized_conll_tokens(data_dir, filenames_for_indexing)
    senna_vecs = get_intersect_vecs(senna_vecs, normalized_conll_tokens)

    vocabulary = senna_vecs.keys()
    voc_int_map = get_vocabulary_int_map(vocabulary)
    senna_int_map = get_features_int_map(senna_vecs, voc_int_map)
    label_int_map = get_label_int_map(data_dir + filenames_for_indexing[-1])
    save_index_by_table(senna_int_map, 'data/embeddings/senna.index')
    save_index_by_table(voc_int_map, data_dir + '/vocab-map.index')
    save_index_by_table(label_int_map, data_dir + '/label-map.index')

    id_map_filename = 'data/conll-abbyy/conll/compreno-vectors.libsvm'
    compreno_id_map = make_compreno_id_map(id_map_filename)

    for filename in filenames:
        print('Processing {0} with {1} mode'.format(filename, mode))
        compreno_line_map = load_compreno_features_mapping(filename)
        compreno_line_to_id_map = make_compreno_line_to_id_map(compreno_line_map, compreno_id_map)

        docs = docs_with_sents(data_dir + filename)
        windowed_docs = windowed_conll_in_id(docs, 1, label_int_map, voc_int_map, compreno_line_to_id_map)
        to_torch, pos = convert_to_torch_format(windowed_docs)

        save_index_by_array(to_torch, data_dir + '/{0}.win.index'.format(filename))
        save_index_by_table(pos, data_dir + '/{0}.win.index-line-pos'.format(filename))

        print(len(windowed_docs), len(to_torch))

def run_sent(mode):
    w2v_file = 'data/embeddings/senna.w2v'
    data_dir = 'data/conll2003/'
    filenames = get_filenames(mode)
    filenames_for_indexing = get_filenames_for_indexing()
    senna_vecs = get_senna_vecs(w2v_file)
    normalized_conll_tokens = get_normalized_conll_tokens(data_dir, filenames_for_indexing)
    senna_vecs = get_intersect_vecs(senna_vecs, normalized_conll_tokens)

    vocabulary = senna_vecs.keys()
    voc_int_map = get_vocabulary_int_map(vocabulary)
    senna_int_map = get_features_int_map(senna_vecs, voc_int_map)
    label_int_map = get_label_int_map(data_dir + filenames_for_indexing[-1])

    save_index_by_table(senna_int_map, 'data/embeddings/senna.index')
    save_index_by_table(voc_int_map, data_dir + '/vocab-map.index')
    save_index_by_table(label_int_map, data_dir + '/label-map.index')

    gazetteers_files = { 'LOC': 'data/gazetteers/ner.loc.lst',
                         'MISC': 'data/gazetteers/ner.misc.lst',
                         'ORG': 'data/gazetteers/ner.org.lst',
                         'PER': 'data/gazetteers/ner.per.lst' }
    gazetteers = init_gazetteers(gazetteers_files)

    for filename in filenames:
        print('Processing {0} with {1} mode'.format(filename, mode))
        docs = docs_with_sents(data_dir + filename)
        sent_docs = sent_conll_in_id(docs, label_int_map, voc_int_map, gazetteers)
        to_torch, pos = convert_to_torch_format(sent_docs)

        save_index_by_array(to_torch, data_dir + '/{0}.index'.format(filename))
        save_index_by_table(pos, data_dir + '/{0}.index-line-pos'.format(filename))

        print(len(sent_docs), len(to_torch))


if __name__ == '__main__':
    mode = argv[1]
    if mode == 'sentence-convolution' or mode == 'compreno-subset':
        run_sent(mode)
    elif mode == 'win' or mode == 'compreno-subset-win':
        run_win(mode)

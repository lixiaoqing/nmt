'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import cPickle as pkl

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

def _translate(seq, tparams, f_init, f_next, options, trng, k, normalize):
    # sample given an input sequence and obtain scores
    sample, score = gen_sample(tparams, f_init, f_next,
                               numpy.array(seq).reshape([len(seq), 1]),
                               options, trng=trng, k=k, maxlen=200,
                               stochastic=False, argmax=False)

    # normalize scores according to sequence lengths
    if normalize:
        lengths = numpy.array([len(s) for s in sample])
        score = score / lengths
    sidx = numpy.argmin(score)
    return sample[sidx]


def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, n_process=5, chr_level=False):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    #init model
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    params = init_params(options)

    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng)

    trans = []
    print 'Translating ', source_file, '...'
    with open(source_file, 'r') as f:
        n = 0
        for line in f:
            n += 1
            if n%10 == 0:
                print n
            if chr_level:
                words = list(line.decode('utf-8').strip())
            else:
                words = line.strip().split()
            x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
            x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
            x += [0]
            y = _translate(x, tparams, f_init, f_next, options, trng, k, normalize)
            trans.append(y)

    with open(saveto, 'w') as f:
        print >>f, '\n'.join(_seqs2words(trans))
    print 'Done'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_target, args.source,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p,
         chr_level=args.c)

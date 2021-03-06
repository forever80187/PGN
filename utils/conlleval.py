from __future__ import division, print_function, unicode_literals

import sys
from collections import defaultdict

def split_tag(chunk_tag):
    
    # if chunk_tag == 'O':
    #     return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_end(prev_tag, tag):
    
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['I'] or prefix1 in ['O']


def is_chunk_start(prev_tag, tag):

    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['I'] or prefix1 in ['O']


def calc_metrics(tp, p, t, percent=True):

    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):

    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'I-O', 'I-O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1
        try:
            _, true_type = split_tag(true_tag)
            _, pred_type = split_tag(pred_tag)
        except:
            print(true_tag, pred_tag)
            continue

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1
    # print(correct_chunk)
    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)


def get_result(correct_chunks, true_chunks, pred_chunks,
               correct_counts, true_counts, pred_counts, verbose=True):

    # sum counts
    
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    # sum_correct_counts = sum(correct_counts.values())
    # sum_true_counts = sum(true_counts.values())

    # nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    # nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    # chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type

    # print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    # print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')

    # print("accuracy: %6.2f%%; (non-O)" % (100 * nonO_correct_counts / nonO_true_counts))
    # print("accuracy: %6.2f%%; " % (100 * sum_correct_counts / sum_true_counts), end='')
    # print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1))

    # # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    # for t in chunk_types:
    #     prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
    #     print("%17s: " % t, end='')
    #     print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
    #           (prec, rec, f1), end='')
    #     print("  %d" % pred_chunks[t])

    # return res


def evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_chunks, true_chunks, pred_chunks,
     correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks,
                        correct_counts, true_counts, pred_counts, verbose=verbose)
    return result


def evaluate_conll_file(fileIterator):
    true_seqs, pred_seqs = [], []

    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return evaluate(true_seqs, pred_seqs)


if __name__ == '__main__':
 
    evaluate_conll_file(sys.stdin)

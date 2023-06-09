#!/usr/bin/env python
# @Project ：NeuralCodeSum 
# @File    ：watch_attns.py
# @Author  ：
# @Date    ：2022/12/12 20:48 
# 
# --------------------------------------------------------
# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

import c2nl.config as config
import c2nl.inputters.utils as util
from c2nl.inputters import constants

from collections import OrderedDict, Counter
from tqdm import tqdm
from c2nl.inputters.timer import AverageMeter, Timer
import c2nl.inputters.vector as vector
import c2nl.inputters.dataset as data

from main.model import Code2NaturalLanguage
from c2nl.eval.bleu import corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor

logger = logging.getLogger()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=200,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=1,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', nargs='+', type=str, default='java',
                       help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='../tmp/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='code',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='../data/10w/',
                       help='Directory of training/validation data')
    files.add_argument('--train_src', nargs='+', type=str, default='train/code_train.txt',
                       help='Preprocessed train source file')
    files.add_argument('--train_src_tag', nargs='+', type=str,
                       help='Preprocessed train source tag file')
    files.add_argument('--train_tgt', nargs='+', type=str, default='train/tgt_train.txt',
                       help='Preprocessed train target file')
    files.add_argument('--dev_src', nargs='+', type=str, default='test/code_test.txt',
                       help='Preprocessed dev source file')
    files.add_argument('--dev_src_tag', nargs='+', type=str,
                       help='Preprocessed dev source tag file')
    files.add_argument('--dev_tgt', nargs='+', type=str, default='test/tgt_test.txt',
                       help='Preprocessed dev target file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=True,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=True,
                            help='Code and summary words will be lower-cased')
    preprocess.add_argument('--src_vocab_size', type=int, default=50000,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=30000,
                            help='Maximum allowed length for tgt dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=True,
                         help='Only do testing')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')

    log.add_argument('--use_commandline', type='bool', default=False)

def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""

    args.dev_src_files = []
    args.dev_tgt_files = []
    args.dev_src_tag_files = []

    num_dataset = 1  # len(args.dataset_name)
    if num_dataset > 1:
        if len(args.dev_src) == 1:
            args.dev_src = args.dev_src * num_dataset
        if len(args.dev_tgt) == 1:
            args.dev_tgt = args.dev_tgt * num_dataset
        if len(args.dev_src_tag) == 1:
            args.dev_src_tag = args.dev_src_tag * num_dataset

    for i in range(num_dataset):
        dataset_name = args.dataset_name
        data_dir = os.path.join(args.data_dir, dataset_name)
        dev_src = os.path.join(data_dir, args.dev_src)
        dev_tgt = os.path.join(data_dir, args.dev_tgt)
        if not os.path.isfile(dev_src):
            raise IOError('No such file: %s' % dev_src)
        if not os.path.isfile(dev_tgt):
            raise IOError('No such file: %s' % dev_tgt)
        if args.use_code_type:
            dev_src_tag = os.path.join(data_dir, args.dev_src_tag)
            if not os.path.isfile(dev_src_tag):
                raise IOError('No such file: %s' % dev_src_tag)
        else:
            dev_src_tag = None

        args.dev_src_files.append(dev_src)
        args.dev_tgt_files.append(dev_tgt)
        args.dev_src_tag_files.append(dev_src_tag)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.pred_file = os.path.join(args.model_dir, args.model_name + suffix + '.json')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    if args.use_src_word or args.use_tgt_word:
        # Make sure fix_embeddings and pretrained are consistent
        if args.fix_embeddings and not args.pretrained:
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    else:
        args.fix_embeddings = False

    return args


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, mode='dev'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()

    with torch.no_grad():
        pbar = tqdm(data_loader)
        for idx, ex in enumerate(pbar):
            if idx == 181:
                batch_size = ex['batch_size']
                ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
                predictions, targets, copy_info = model.predict(ex, replace_unk=True)

                src_sequences = [code for code in ex['code_text']]
                examples += batch_size
                for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                    hypotheses[key] = [pred]
                    references[key] = tgt if isinstance(tgt, list) else [tgt]
                    sources[key] = src
                # print(hypotheses)

                if copy_info is not None:
                    copy_info = copy_info.cpu().numpy().astype(int).tolist()
                    for key, cp in zip(ex_ids, copy_info):
                        copy_dict[key] = cp

                pbar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])


    copy_dict = None if len(copy_dict) == 0 else copy_dict

    print('%' * 50)
    hy = [h[0] for h in list(hypotheses.values())]
    re = [h[0] for h in list(references.values())]
    print(hy[0])
    from main.evaluation import nltk_bleu
    nltk_bleu(hy, re, 'fff')

    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=args.pred_file,
                                                                   print_copy_info=args.print_copy_info,
                                                                   mode=mode)
    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    if mode == 'test':
        logger.info('test valid official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())

    else:
        logger.info('dev valid official: Epoch = %d | ' %
                    (global_stats['epoch']) +
                    'bleu = %.2f | rouge_l = %.2f | '
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
                    (bleu, rouge_l, precision, recall, f1, examples) +
                    'valid time = %.2f (s)' % eval_time.time())

    return result


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1


def eval_accuracies(hypotheses, references, copy_info, sources=None,
                    filename=None, print_copy_info=False, mode='dev'):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, ind_bleu = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
                                              references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        if fw:
            if copy_info is not None and print_copy_info:
                prediction = hypotheses[key][0].split()
                pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
                          for j, word in enumerate(prediction)]
                pred_i = [' '.join(pred_i)]
            else:
                pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            logobj['references'] = references[key][0] if args.print_one_target \
                else references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            fw.write(json.dumps(logobj) + '\n')

    if fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')

    dev_exs = []
    dev_src, dev_src_tag, dev_tgt, dataset_name = args.dev_src_files[0], args.dev_src_tag_files[0], args.dev_tgt_files[0], args.dataset_name
    print(dev_src, dev_src_tag, dev_tgt, dataset_name)
    dev_files = dict()
    dev_files['src'] = dev_src
    dev_files['src_tag'] = dev_src_tag
    dev_files['tgt'] = dev_tgt
    exs = util.load_data(args,
                         dev_files,
                         max_examples=args.max_examples,
                         dataset_name=dataset_name,
                         test_split=True)
    dev_exs.extend(exs)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if args.pretrained:
        model = Code2NaturalLanguage.load(args.pretrained)
    else:
        if not os.path.isfile(args.model_file):
            raise IOError('No such file: %s' % args.model_file)
        model = Code2NaturalLanguage.load(args.model_file)

    # Use the GPU?
    if args.cuda:
        model.cuda()

    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    dev_dataset = data.CommentDataset(dev_exs, model)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
    validate_official(args, dev_loader, model, stats, mode='test')



if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)

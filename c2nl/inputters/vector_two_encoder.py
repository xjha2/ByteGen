import torch


def vectorize(ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict
    cfg_dict = model.cfg_dict
    tgt_dict = model.tgt_dict

    code, cfg, summary = ex['code'], ex['cfg'], ex['summary']
    vectorized_ex = dict()
    vectorized_ex['id'] = code.id
    vectorized_ex['language'] = code.language

    vectorized_ex['code'] = code.text
    vectorized_ex['code_tokens'] = code.tokens
    vectorized_ex['code_char_rep'] = None
    vectorized_ex['code_type_rep'] = None
    vectorized_ex['code_mask_rep'] = None
    vectorized_ex['use_code_mask'] = False

    vectorized_ex['code_word_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict))
    if model.args.use_src_char:
        vectorized_ex['code_char_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict, _type='char'))
    if model.args.use_code_type:
        vectorized_ex['code_type_rep'] = torch.LongTensor(code.type)
    if code.mask:
        vectorized_ex['code_mask_rep'] = torch.LongTensor(code.mask)
        vectorized_ex['use_code_mask'] = True

    vectorized_ex['cfg'] = cfg.text
    vectorized_ex['cfg_tokens'] = cfg.tokens
    vectorized_ex['cfg_char_rep'] = None
    vectorized_ex['cfg_type_rep'] = None
    vectorized_ex['cfg_mask_rep'] = None
    vectorized_ex['use_cfg_mask'] = False

    vectorized_ex['cfg_word_rep'] = torch.LongTensor(cfg.vectorize(word_dict=cfg_dict))
    if model.args.use_cfg_char:
        vectorized_ex['cfg_char_rep'] = torch.LongTensor(cfg.vectorize(word_dict=cfg_dict, _type='char'))
    if model.args.use_cfg_type:
        vectorized_ex['cfg_type_rep'] = torch.LongTensor(cfg.type)
    if cfg.mask:
        vectorized_ex['cfg_mask_rep'] = torch.LongTensor(cfg.mask)
        vectorized_ex['use_cfg_mask'] = True

    vectorized_ex['summ'] = None
    vectorized_ex['summ_tokens'] = None
    vectorized_ex['stype'] = None
    vectorized_ex['summ_word_rep'] = None
    vectorized_ex['summ_char_rep'] = None
    vectorized_ex['target'] = None

    if summary is not None:
        vectorized_ex['summ'] = summary.text
        vectorized_ex['summ_tokens'] = summary.tokens
        vectorized_ex['stype'] = summary.type
        vectorized_ex['summ_word_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict))
        if model.args.use_tgt_char:
            vectorized_ex['summ_char_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict, _type='char'))
        # target is only used to compute loss during training
        vectorized_ex['target'] = torch.LongTensor(summary.vectorize(tgt_dict))

    vectorized_ex['src_vocab'] = code.src_vocab
    vectorized_ex['cfg_vocab'] = cfg.src_vocab
    vectorized_ex['use_src_word'] = model.args.use_src_word
    vectorized_ex['use_cfg_word'] = model.args.use_cfg_word
    vectorized_ex['use_tgt_word'] = model.args.use_tgt_word
    vectorized_ex['use_src_char'] = model.args.use_src_char
    vectorized_ex['use_cfg_char'] = model.args.use_cfg_char
    vectorized_ex['use_tgt_char'] = model.args.use_tgt_char
    vectorized_ex['use_code_type'] = model.args.use_code_type
    vectorized_ex['use_cfg_type'] = model.args.use_cfg_type

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]['use_src_word']
    use_cfg_word = batch[0]['use_cfg_word']
    use_tgt_word = batch[0]['use_tgt_word']
    use_src_char = batch[0]['use_src_char']
    use_cfg_char = batch[0]['use_cfg_char']
    use_tgt_char = batch[0]['use_tgt_char']
    use_code_type = batch[0]['use_code_type']
    use_code_mask = batch[0]['use_code_mask']
    use_cfg_type = batch[0]['use_cfg_type']
    use_cfg_mask = batch[0]['use_cfg_mask']
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]

    # --------- Prepare Code tensors ---------
    code_words = [ex['code_word_rep'] for ex in batch]
    code_chars = [ex['code_char_rep'] for ex in batch]
    code_type = [ex['code_type_rep'] for ex in batch]
    code_mask = [ex['code_mask_rep'] for ex in batch]
    max_code_len = max([d.size(0) for d in code_words])
    if use_src_char:
        max_char_in_code_token = code_chars[0].size(1)

    # Batch Code Representations
    code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_src_word else None
    code_type_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_type else None
    code_mask_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_mask else None
    code_char_rep = torch.zeros(batch_size, max_code_len, max_char_in_code_token, dtype=torch.long) \
        if use_src_char else None

    source_maps = []
    src_vocabs = []
    for i in range(batch_size):
        code_len_rep[i] = code_words[i].size(0)
        if use_src_word:
            code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
        if use_code_type:
            code_type_rep[i, :code_type[i].size(0)].copy_(code_type[i])
        if use_code_mask:
            code_mask_rep[i, :code_mask[i].size(0)].copy_(code_mask[i])
        if use_src_char:
            code_char_rep[i, :code_chars[i].size(0), :].copy_(code_chars[i])
        #
        context = batch[i]['code_tokens']
        vocab = batch[i]['src_vocab']
        src_vocabs.append(vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([vocab[w] for w in context])
        source_maps.append(src_map)

    # --------- Prepare cfg tensors ---------
    cfg_words = [ex['cfg_word_rep'] for ex in batch]
    cfg_chars = [ex['cfg_char_rep'] for ex in batch]
    cfg_type = [ex['cfg_type_rep'] for ex in batch]
    cfg_mask = [ex['cfg_mask_rep'] for ex in batch]
    max_cfg_len = max([d.size(0) for d in cfg_words])
    if use_cfg_char:
        max_char_in_cfg_token = cfg_chars[0].size(1)

    # Batch Code Representations
    cfg_len_rep = torch.zeros(batch_size, dtype=torch.long)
    cfg_word_rep = torch.zeros(batch_size, max_cfg_len, dtype=torch.long) \
        if use_cfg_word else None
    cfg_type_rep = torch.zeros(batch_size, max_cfg_len, dtype=torch.long) \
        if use_cfg_type else None
    cfg_mask_rep = torch.zeros(batch_size, max_cfg_len, dtype=torch.long) \
        if use_cfg_mask else None
    cfg_char_rep = torch.zeros(batch_size, max_cfg_len, max_char_in_cfg_token, dtype=torch.long) \
        if use_cfg_char else None

    cfg_maps = []
    cfg_vocabs = []
    for i in range(batch_size):
        cfg_len_rep[i] = cfg_words[i].size(0)
        if use_cfg_word:
            cfg_word_rep[i, :cfg_words[i].size(0)].copy_(cfg_words[i])
        if use_cfg_type:
            cfg_type_rep[i, :cfg_type[i].size(0)].copy_(cfg_type[i])
        if use_cfg_mask:
            cfg_mask_rep[i, :cfg_mask[i].size(0)].copy_(cfg_mask[i])
        if use_cfg_char:
            cfg_char_rep[i, :cfg_chars[i].size(0), :].copy_(cfg_chars[i])
        #
        cfg_context = batch[i]['cfg_tokens']
        cfg_vocab = batch[i]['cfg_vocab']
        cfg_vocabs.append(cfg_vocab)
        # Mapping source tokens to indices in the dynamic dict.
        cfg_map = torch.LongTensor([cfg_vocab[w] for w in cfg_context])
        cfg_maps.append(cfg_map)

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summ_word_rep'] is None
    if no_summary:
        summ_len_rep = None
        summ_word_rep = None
        summ_char_rep = None
        tgt_tensor = None
        alignments = None
    else:
        summ_words = [ex['summ_word_rep'] for ex in batch]
        summ_chars = [ex['summ_char_rep'] for ex in batch]
        max_sum_len = max([q.size(0) for q in summ_words])
        if use_tgt_char:
            max_char_in_summ_token = summ_chars[0].size(1)

        summ_len_rep = torch.zeros(batch_size, dtype=torch.long)
        summ_word_rep = torch.zeros(batch_size, max_sum_len, dtype=torch.long) \
            if use_tgt_word else None
        summ_char_rep = torch.zeros(batch_size, max_sum_len, max_char_in_summ_token, dtype=torch.long) \
            if use_tgt_char else None

        max_tgt_length = max([ex['target'].size(0) for ex in batch])
        tgt_tensor = torch.zeros(batch_size, max_tgt_length, dtype=torch.long)
        alignments = []
        for i in range(batch_size):
            summ_len_rep[i] = summ_words[i].size(0)
            if use_tgt_word:
                summ_word_rep[i, :summ_words[i].size(0)].copy_(summ_words[i])
            if use_tgt_char:
                summ_char_rep[i, :summ_chars[i].size(0), :].copy_(summ_chars[i])
            #
            tgt_len = batch[i]['target'].size(0)
            tgt_tensor[i, :tgt_len].copy_(batch[i]['target'])
            target = batch[i]['summ_tokens']
            align_mask = torch.LongTensor([src_vocabs[i][w] for w in target])
            alignments.append(align_mask)

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_char_rep': code_char_rep,
        'code_type_rep': code_type_rep,
        'code_mask_rep': code_mask_rep,
        'code_len': code_len_rep,
        'cfg_word_rep': cfg_word_rep,
        'cfg_char_rep': cfg_char_rep,
        'cfg_type_rep': cfg_type_rep,
        'cfg_mask_rep': cfg_mask_rep,
        'cfg_len': cfg_len_rep,
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'summ_len': summ_len_rep,
        'tgt_seq': tgt_tensor,
        'code_text': [ex['code'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'cfg_text': [ex['cfg'] for ex in batch],
        'cfg_tokens': [ex['cfg_tokens'] for ex in batch],
        'summ_text': [ex['summ'] for ex in batch],
        'summ_tokens': [ex['summ_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'cfg_vocab': cfg_vocabs,
        'cfg_map': cfg_maps,
        'alignment': alignments,
        'stype': [ex['stype'] for ex in batch]
    }

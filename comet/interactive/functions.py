import torch
import sys
import os
import time

from comet.data.utils import TextEncoder
import comet.data.config as cfg
import comet.data.data as data
import comet.models.models as models
from comet.evaluate.sampler import BeamSampler, GreedySampler, TopKSampler

import comet.utils as utils


def set_compute_mode(mode):
    if mode == "auto" and torch.cuda.is_available():
        cfg.device = "cuda"
    elif mode.isdigit():
        cfg.device = int(mode)
    elif mode in ["cpu", "cuda"]:
        cfg.device = mode
    else:
        raise
    print("Pushing units to: {}".format(cfg.device))


class BaseDataLoader(object):
    def __init__(self, opt, vocab):
        self.vocab_encoder = vocab
        self.vocab_decoder = {j: i for i, j in self.vocab_encoder.items()}


class ConceptNetBaseDataLoader(BaseDataLoader):
    def __init__(self, opt, vocab):
        super(ConceptNetBaseDataLoader, self).__init__(opt, vocab)
        self.max_e1 = opt.data.get("maxe1", 10)
        self.max_e2 = opt.data.get("maxe2", 15) + 1
        self.max_r = opt.data.get("maxr", 5)


class AtomicBaseDataLoader(BaseDataLoader):
    def __init__(self, opt, vocab):
        super(AtomicBaseDataLoader, self).__init__(opt, vocab)
        self.max_event = opt.data.get("maxe1", 17)
        self.max_effect = opt.data.get("maxe2", 34) + 1


def load_model_file(model_file):
    model_stuff = data.load_checkpoint(model_file)
    opt = utils.convert_nested_dict_to_DD(model_stuff["opt"])

    state_dict = model_stuff["state_dict"]
    vocab = model_stuff['vocab']
    return opt, state_dict, vocab


def load_data(dataset, opt, vocab, vocabulary_path):
    if dataset == "atomic":
        data_loader = load_atomic_data(opt, vocab)
    elif dataset == "conceptnet":
        data_loader = load_conceptnet_data(opt, vocab)

    # Initialize TextEncoder
    encoder_path = vocabulary_path + "encoder_bpe_40000.json"
    bpe_path = vocabulary_path + "vocab_40000.bpe"
    text_encoder = TextEncoder(encoder_path, bpe_path)
    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    return data_loader, text_encoder


def load_atomic_data(opt, vocab):
    # path = "data/atomic/processed/generation/{}.pickle".format(
    #     utils.make_name_string(opt.data))
    # data_loader = data.make_data_loader(opt, opt.data.categories)
    # loaded = data_loader.load_data(path)

    data_loader = AtomicBaseDataLoader(opt, vocab)

    return data_loader


def load_conceptnet_data(opt, vocab):
    # path = "data/conceptnet/processed/generation/{}.pickle".format(
    # utils.make_name_string(opt.data))
    # data_loader = data.make_data_loader(opt)
    # loaded = data_loader.load_data(path)
    data_loader = ConceptNetBaseDataLoader(opt, vocab)
    return data_loader


def make_model(opt, n_vocab, n_ctx, state_dict):
    model = models.make_model(
        opt, n_vocab, n_ctx, None, load=False,
        return_acts=True, return_probs=False)
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    return model


def set_sampler(opt, sampling_algorithm, data_loader):
    if "beam" in sampling_algorithm:
        opt.eval.bs = int(sampling_algorithm.split("-")[1])
        sampler = BeamSampler(opt, data_loader)
    elif "topk" in sampling_algorithm:
        # print("Still bugs in the topk sampler. Use beam or greedy instead")
        # raise NotImplementedError
        opt.eval.k = int(sampling_algorithm.split("-")[1])
        sampler = TopKSampler(opt, data_loader)
    else:
        sampler = GreedySampler(opt, data_loader)

    return sampler


def get_atomic_sequence(input_event, model, sampler, data_loader, text_encoder, category):
    if isinstance(category, list):
        outputs = {}
        for cat in category:
            new_outputs = get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, cat)
            outputs.update(new_outputs)
        return outputs
    elif category == "all":
        outputs = {}

        # start = time.time()

        for category in data.atomic_data.all_categories:
            new_outputs = get_atomic_sequence(
                input_event, model, sampler, data_loader, text_encoder, category)
            outputs.update(new_outputs)

        # end = time.time()
        # print("total time for all categories: {} s".format(end - start))
        return outputs
    else:

        sequence_all = {}

        sequence_all["event"] = input_event
        sequence_all["effect_type"] = category

        with torch.no_grad():

            # start = time.time()

            batch = set_atomic_inputs(
                input_event, category, data_loader, text_encoder)

            # end_set = time.time()

            sampling_result = sampler.generate_sequence(
                batch, model, data_loader, data_loader.max_event +
                data.atomic_data.num_delimiter_tokens["category"],
                data_loader.max_effect -
                data.atomic_data.num_delimiter_tokens["category"])

            # end_sample = time.time()
            # print(category)
            # print("Set inputs: {} s".format(end_set - start))
            # print("Sample: {} s".format(end_sample - end_set))

        sequence_all['beams'] = sampling_result["beams"]

        print_atomic_sequence(sequence_all)

        return {category: sequence_all}


def print_atomic_sequence(sequence_object):
    input_event = sequence_object["event"]
    category = sequence_object["effect_type"]

    print("Input Event:   {}".format(input_event))
    print("Target Effect: {}".format(category))
    print("")
    print("Candidate Sequences:")
    for beam in sequence_object["beams"]:
        print(beam)
    print("")
    print("====================================================")
    print("")


def set_atomic_inputs(input_event, category, data_loader, text_encoder):
    XMB = torch.zeros(1, data_loader.max_event + 1).long().to(cfg.device)
    prefix, suffix = data.atomic_data.do_example(text_encoder, input_event, None, True, None)

    XMB[:, :len(prefix)] = torch.LongTensor(prefix)
    XMB[:, -1] = torch.LongTensor([text_encoder.encoder["<{}>".format(category)]])

    batch = {}
    batch["sequences"] = XMB
    batch["attention_mask"] = data.atomic_data.make_attention_mask(XMB)

    return batch


def get_conceptnet_sequence(e1, model, sampler, data_loader, text_encoder, relation, force=False):
    if isinstance(relation, list):
        outputs = {}

        for rel in relation:
            new_outputs = get_conceptnet_sequence(
                e1, model, sampler, data_loader, text_encoder, rel)
            outputs.update(new_outputs)
        return outputs
    elif relation == "all":
        outputs = {}

        for relation in data.conceptnet_data.conceptnet_relations:
            new_outputs = get_conceptnet_sequence(
                e1, model, sampler, data_loader, text_encoder, relation)
            outputs.update(new_outputs)
        return outputs
    else:

        sequence_all = {}

        sequence_all["e1"] = e1
        sequence_all["relation"] = relation

        with torch.no_grad():
            if data_loader.max_r != 1:
                relation_sequence = data.conceptnet_data.split_into_words[relation]
            else:
                relation_sequence = "<{}>".format(relation)

            batch, abort = set_conceptnet_inputs(
                e1, relation_sequence, text_encoder,
                data_loader.max_e1, data_loader.max_r, force)

            if abort:
                return {relation: sequence_all}

            sampling_result = sampler.generate_sequence(
                batch, model, data_loader,
                data_loader.max_e1 + data_loader.max_r,
                data_loader.max_e2)

        sequence_all['beams'] = sampling_result["beams"]

        print_conceptnet_sequence(sequence_all)

        return {relation: sequence_all}


def set_conceptnet_inputs(input_event, relation, text_encoder, max_e1, max_r, force):
    abort = False

    e1_tokens, rel_tokens, _ = data.conceptnet_data.do_example(text_encoder, input_event, relation, None)

    if len(e1_tokens) >  max_e1:
        if force:
            XMB = torch.zeros(1, len(e1_tokens) + max_r).long().to(cfg.device)
        else:
            XMB = torch.zeros(1, max_e1 + max_r).long().to(cfg.device)
            return {}, True
    else:
        XMB = torch.zeros(1, max_e1 + max_r).long().to(cfg.device)

    XMB[:, :len(e1_tokens)] = torch.LongTensor(e1_tokens)
    XMB[:, max_e1:max_e1 + len(rel_tokens)] = torch.LongTensor(rel_tokens)

    batch = {}
    batch["sequences"] = XMB
    batch["attention_mask"] = data.conceptnet_data.make_attention_mask(XMB)

    return batch, abort


def print_conceptnet_sequence(sequence_object):
    e1 = sequence_object["e1"]
    relation = sequence_object["relation"]

    print("Input Entity:    {}".format(e1))
    print("Target Relation: {}".format(relation))
    print("")
    print("Candidate Sequences:")
    for beam in sequence_object["beams"]:
        print(beam)
    print("")
    print("====================================================")
    print("")


def print_help(data):
    print("")
    if data == "atomic":
        print("Provide a seed event such as \"PersonX goes to the mall\"")
        print("Don't include names, instead replacing them with PersonX, PersonY, etc.")
        print("The event should always have PersonX included")
    if data == "conceptnet":
        print("Provide a seed entity such as \"go to the mall\"")
        print("Because the model was trained on lemmatized entities,")
        print("it works best if the input entities are also lemmatized")
    print("")


def print_relation_help(data):
    print_category_help(data)


def print_category_help(data):
    print("")
    if data == "atomic":
        print("Enter a possible effect type from the following effect types:")
        print("all - compute the output for all effect types {{oEffect, oReact, oWant, xAttr, xEffect, xIntent, xNeed, xReact, xWant}}")
        print("oEffect - generate the effect of the event on participants other than PersonX")
        print("oReact - generate the reactions of participants other than PersonX to the event")
        print("oEffect - generate what participants other than PersonX may want after the event")
    elif data == "conceptnet":
        print("Enter a possible relation from the following list:")
        print("")
        print('AtLocation')
        print('CapableOf')
        print('Causes')
        print('CausesDesire')
        print('CreatedBy')
        print('DefinedAs')
        print('DesireOf')
        print('Desires')
        print('HasA')
        print('HasFirstSubevent')
        print('HasLastSubevent')
        print('HasPainCharacter')
        print('HasPainIntensity')
        print('HasPrerequisite')
        print('HasProperty')
        print('HasSubevent')
        print('InheritsFrom')
        print('InstanceOf')
        print('IsA')
        print('LocatedNear')
        print('LocationOfAction')
        print('MadeOf')
        print('MotivatedByGoal')
        print('NotCapableOf')
        print('NotDesires')
        print('NotHasA')
        print('NotHasProperty')
        print('NotIsA')
        print('NotMadeOf')
        print('PartOf')
        print('ReceivesAction')
        print('RelatedTo')
        print('SymbolOf')
        print('UsedFor')
        print("")
        print("NOTE: Capitalization is important")
    else:
        raise
    print("")

def print_sampling_help():
    print("")
    print("Provide a sampling algorithm to produce the sequence with from the following:")
    print("")
    print("greedy")
    print("beam-# where # is the beam size")
    print("topk-# where # is k")
    print("")

import sys
import os
import argparse

import comet.interactive.functions as interactive

sampling_mapping = {
    "b10": "beam-10",
    "b5": "beam-5",
    "g": "greedy"
}

def parse_input_string(string):
    objects = string.split("|")
    relations = objects[1]

    if not relations or relations == "all":
        base_relations = [
            'AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
            'CreatedBy', 'DefinedAs', 'Desires', 'HasA', 'HasFirstSubevent',
            'HasLastSubevent', 'HasPrerequisite', 'HasProperty', 'HasSubevent',
            'IsA', 'MadeOf', 'MotivatedByGoal', 'PartOf', 'ReceivesAction',
            'SymbolOf', 'UsedFor'
        ]
        # final_relations = ["<{}>".format(i) for i in base_relations]
    else:
        final_relations = relations.split(",")

    sampling = sampling_mapping[objects[2]]
    sequence = objects[0]

    return sequence, final_relations, sampling


def format_output_string(text_sequence, sequences):
    print_string = []

    print_string.append("<h3>{}</h3>".format(text_sequence))

    for relation, stuff in sequences.items():
        print_string.append("<b>{}</b>".format(relation))
        for i, sequence in enumerate(stuff["beams"]):
            print_string.append("({}) {}".format(i + 1, sequence))
        print_string.append("")
    print_string.append("")
    return "<br>".join(print_string)


class DemoModel(object):
    def __init__(self, model_file, vocabulary_path="model/"):
        opt, state_dict, vocab = interactive.load_model_file(model_file)

        data_loader, text_encoder = interactive.load_data(
            "conceptnet", opt, vocab, vocabulary_path)

        self.opt = opt
        self.data_loader = data_loader
        self.text_encoder = text_encoder

        n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
        n_vocab = len(text_encoder.encoder) + n_ctx

        model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

        self.model = model


    def predict(self, text_sequence, relations, sampling_algorithm, verbose=True):
        sampler = interactive.set_sampler(
            self.opt, sampling_algorithm, self.data_loader)

        sequences = interactive.get_conceptnet_sequence(
            text_sequence, self.model, sampler, self.data_loader,
            self.text_encoder, relations)

        return sequences

    def getOutput(self, text_string):
        text_sequence, relations, sampling_algorithm = parse_input_string(text_string)

        model_output_sequences = self.predict(
            text_sequence, relations, sampling_algorithm, verbose=True)

        return format_output_string(text_sequence, model_output_sequences)


if __name__ == "__main__":
    # sys.path.append("ConceptNet_NLGWebsite")
    # sys.path.append(os.getcwd())
    from server import run

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_15500.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()

    interactive.set_compute_mode(args.device)

    myNLGmodel = DemoModel(args.model_file)

    run(nlg=myNLGmodel, port=8001)

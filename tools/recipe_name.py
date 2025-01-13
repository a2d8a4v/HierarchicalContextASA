from tools.utils import BERT2ABB

def get_recipe_name(hps):

    if hps.save_dir_name is None:
        hps.save_dir_name = "{}".format(
            '.'.join(
                [hps.model, hps.sentaspara] + \
                [str(hps.lr)] + \
                (['reweight{}'.format('' if hps.rw_alpha == 1. else hps.rw_alpha)] if hps.reweight else []) + \
                (["{}{}".format(hps.imp, hps.init_noise_sigma)] if hps.imp else []) + \
                [hps.problem_type] + \
                [hps.head] + \
                ([] if hps.pred_method is None else [hps.pred_method]) + \
                ([BERT2ABB[hps.bert_model_path]] if hps.bert else []) + \
                (['bt'] if hps.bert_train else ['peft'] if hps.bert_train_peft else ['ft'] if hps.bert_train_finetune else []) + \
                (['glove'] if hps.word_embedding else ['randembed']) + \
                (['pmi{}'.format(hps.pmi_window_width)] if hps.pmi_window_width > -1 else []) + \
                (['interviewer'] if hps.interviewer else []) + \
                (['wpr'] if hps.retain_wp_relation else []) + \
                (['rmp'] if hps.revserse_metapath else []) + \
                (['cefr{}'.format(hps.cefr_info)] if hps.cefr_word else []) + \
                (['fp{}'.format(hps.filled_pauses_info)] if hps.filled_pauses_word else []) + \
                (['oe'] if hps.oe and hps.oe_weight == 1e-3 else ['oe{}'.format(hps.oe_weight)] if hps.oe and hps.oe_weight != 1e-3 else [] ) + \
                (['wcefr'] if hps.wcefr else []) + \
                (['wr'] if hps.wcefr_reweight else []) + \
                (['han_s'] if hps.han_s else []) + \
                (['lu'] if hps.language_use else []) + \
                (['memn2n'] if hps.memn2n else [])
            )
        )
    if hps.baseline:
        hps.save_dir_name = "{}".format(
            '.'.join(
                [hps.model, hps.sentaspara] + \
                [str(hps.lr)] + \
                (['reweight{}'.format('' if hps.rw_alpha == 1. else hps.rw_alpha)] if hps.reweight else []) + \
                (["{}{}".format(hps.imp, hps.init_noise_sigma)] if hps.imp else []) + \
                [hps.problem_type] + \
                [hps.head] + \
                ([BERT2ABB[hps.bert_model_path]] if hps.bert else []) + \
                (['bt'] if hps.bert_train else ['peft'] if hps.bert_train_peft else ['ft'] if hps.bert_train_finetune else []) + \
                ['baseline'] + \
                (['lu'] if hps.language_use else []) + \
                (['memn2n'] if hps.memn2n else [])
            )
        )
        
    return hps
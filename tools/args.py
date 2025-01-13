import argparse
from distutils.util import strtobool as dist_strtobool
import configargparse

def strtobool(x):
    # distutils.util.strtobool returns integer, but it's confusing,
    return bool(dist_strtobool(x))


## Train Arguments
def get_train_args(aux_objs):
    
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM',help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CNNDM',help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--model', type=str, default='HSG', help='model structure[HSG|HDSG]')
    parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')
    parser.add_argument('--save_dir_name', type=str, default=None, help='Root directory for all logging.')

    # Speaker-wise
    parser.add_argument('--train_speaker_wise', action='store_true', default=False, help='Run training in speaker-wise.')
    parser.add_argument('--eval_speaker_wise', action='store_true', default=False, help='Run evaluation in terms of speaker-wise results.')

    # Hyperparameters
    parser.add_argument('--seed', type=int, default=666, help='set the random seed [default: 666]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--bert_gpu', type=int, default=1, help='GPU ID of BERT model to use. [default: 4]')
    parser.add_argument('--cuda', action='store_true', default=False, help='GPU or CPU [default: False]')
    parser.add_argument('--num_workers', type=int, default=0, help='numbers of workers [default: 32]')
    parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=4, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')
    parser.add_argument('--reweight', action='store_true', default=False, help='Reweight the loss when training and evaluation')
    parser.add_argument('--rw_alpha', type=float, default=1.5, help='iteration hop [default: 1.5]')

    parser.add_argument('--word_embedding', action='store_true', default=False, help='whether to use Word embedding [default: False]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False,help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size [default: 64]')
    parser.add_argument('--spo_hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--pmi_window_width', type=int, default=-1,help='Use PMI information for word node to word node')
    parser.add_argument('--sent_max_len', type=int, default=1300,help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--passage_length', type =int , default=10)
    parser.add_argument('--doc_max_timesteps', type=int, default=10,help='max length of documents (max timesteps of documents)')
    parser.add_argument('--revserse_metapath', action='store_true', default=False, help='Revserse the meta path of updating processing [default: False]')
    parser.add_argument('--retain_wp_relation', action='store_true', default=False, help='Revserse the meta path of updating processing [default: False]')
    parser.add_argument('--head', type=str, default='linear', choices=['linear', 'predictionhead'], help="Prediction Head")

    # BERT fusion
    parser.add_argument('--bert', action='store_true', default=False, help='BERT embedding fusion')
    parser.add_argument('--bert_model_path', type=str, default="roberta-base", choices=['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'allenai/longformer-base-4096', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b'], help="Pre-trained BERT model to extend. e.g. ['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b]")
    parser.add_argument('--bert_train', action='store_true', default=False, help='BERT do not freeze')
    parser.add_argument('--bert_train_peft', action='store_true', default=False, help='BERT do not freeze, but using PEFT')
    parser.add_argument('--bert_train_finetune', action='store_true', default=False, help='BERT do not freeze, but using finetuned BERT')
    parser.add_argument('--bert_roberta_to_long', action='store_true', default=False, help='Using this option to convert the Transformer Encoder from roberta with changeable max pos length, not directly from Longformer')
    
    # Interviewer information
    parser.add_argument('--interviewer', action='store_true', default=False, help='Use interviewer information')
    
    # information embedding
    parser.add_argument('--cefr_word', action='store_true', default=False, help='Use CEFR vocabulary profile information')
    parser.add_argument('--cefr_info', type=str, default=None, choices=[None, 'embed_init', 'graph_init'], help="CEFR node embedding")
    parser.add_argument('--filled_pauses_word', action='store_true', default=False, help='Use disfluency tag information')
    parser.add_argument('--filled_pauses_info', type=str, default="embed_init", choices=['embed_init', 'graph_init'], help="Filled Pause embedding")

    # debug
    parser.add_argument('--final_attention', action='store_true', default=False, help='Use CEFR vocabulary profile information')

    # ordinal entropy
    parser.add_argument('--oe', action='store_true', default=False, help='Use ordinal entropy')
    parser.add_argument('--oe_weight', type=float, default=1e-3, help='learning rate')

    # han in sentence to paragraph level feature
    parser.add_argument('--han_s', action='store_true', default=False, help='Use CEFR vocabulary profile information')

    # use document as features
    parser.add_argument('--use_doc', action='store_true', default=False, help='whether use doc or not')

    # Word CEFR as a new training objective
    parser.add_argument('--wcefr', action='store_true', default=False, help='add word cefr as training objective')
    parser.add_argument('--wcefr_reweight', action='store_true', default=False, help='Rewieght on word cefr loss')

    # cheat baseline
    parser.add_argument('--baseline', action='store_true', default=False, help='Use ordinal entropy')

    # other experiments
    parser.add_argument('--language_use', action='store_true', default=False, help='Only train on Role-a-paly scenario')
    parser.add_argument('--memn2n', action='store_true', default=False, help='Only train on Role-a-paly scenario')

    # deep regression
    parser.add_argument('--bmse', action='store_true', default=False, help='use Balanced MSE')
    parser.add_argument('--imp', type=str, default=None, choices=[None, 'gai', 'bmc', 'bni'], help='implementation options')
    parser.add_argument('--gmm', type=str, default='gmm.pkl', help='path to preprocessed GMM')
    parser.add_argument('--init_noise_sigma', type=float, default=0.85, help='initial scale of the noise')
    parser.add_argument('--sigma_lr', type=float, default=1e-2, help='learning rate of the noise scale')
    parser.add_argument('--fix_noise_sigma', action='store_true', default=False, help='disable joint optimization')

    parser.add_argument('--verbose_show_in_gpumonitor', type=str, default='Its usage on GPU memory range sharpely from 3 GB to 20GB', help='Its usage on GPU memory range sharpely from 3 GB to 20GB')

    # Auxiliary objectives
    for obj in aux_objs:
        parser.add_argument('--use_{}'.format(obj), action='store_true', default=False,
                            help='Use {} objective during training.'.format(obj, obj))

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')
    parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='for gradient clipping max gradient normalization')
    parser.add_argument('--problem_type', type=str, default='classification', choices=['regression', 'classification', 'r2'], help='Regard problem as regression classification')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--non_descent_count', type=int, default=4, help='Early stop if loss not getting lower within specific times')
    parser.add_argument('--wandb', action='store_true', default=False, help='WanDB to record your training states')
    parser.add_argument('--pred_method', type=str, default=None, choices=[None, 'hec', 'ehg', 'acg', 'hsag', 'bert', 'sde', 'all_s', 'sde_s', 'hec_s', 'test_wdc', 'test_wdc2', 'test_wdc3'],help='max length of documents (max timesteps of documents)')
    parser.add_argument('--stdout_metric', action='store_true', default=False, help='Print metrics during training')

    parser.add_argument('--save_gpu_mode', action='store_true', default=False, help='Save GPU when no enough vram')

    args = parser.parse_args()
    return args


## Evaluation Arguments
def get_eval_args(aux_objs):
    
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM', help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CNNDM', help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--model', type=str, default="HSumGraph", help="model structure[HSG|HDSG]")
    parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')
    parser.add_argument('--save_dir_name', type=str, default=None, help='Root directory for all logging.')

    # Speaker-wise
    parser.add_argument('--train_speaker_wise', action='store_true', default=False, help='Run training in speaker-wise.')
    parser.add_argument('--eval_speaker_wise', action='store_true', default=True, help='Run evaluation in terms of speaker-wise results.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--bert_gpu', type=int, default=1, help='GPU ID of BERT model to use. [default: 4]')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--num_workers', type=int, default=0, help='numbers of workers [default: 32]')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary.')
    parser.add_argument('--batch_size', type=int, default=4, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration ')
    parser.add_argument('--reweight', action='store_true', default=False, help='Reweight the loss when training and evaluation')
    parser.add_argument('--rw_alpha', type=float, default=1.5, help='iteration hop [default: 1.5]')

    # Arguments when training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

    # BERT fusion
    parser.add_argument('--bert', action='store_true', default=False, help='BERT embedding fusion')
    parser.add_argument('--bert_model_path', type=str, default="roberta-base", choices=['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'allenai/longformer-base-4096', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b'], help="Pre-trained BERT model to extend. e.g. ['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b]")
    parser.add_argument('--bert_train', action='store_true', default=False, help='Mean Pooling After BERT encoding')
    parser.add_argument('--bert_train_peft', action='store_true', default=False, help='BERT do not freeze, but using PEFT')
    parser.add_argument('--bert_train_finetune', action='store_true', default=False, help='BERT do not freeze, but using finetuned BERT')
    parser.add_argument('--bert_roberta_to_long', action='store_true', default=False, help='Using this option to convert the Transformer Encoder from roberta with changeable max pos length, not directly from Longformer')

    parser.add_argument('--word_embedding', action='store_true', default=False, help='whether to use Word embedding')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False, help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state')
    parser.add_argument('--lstm_layers', type=int, default=2, help='lstm layers')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='use bidirectional LSTM')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size [default: 64]')
    parser.add_argument('--spo_hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--gcn_hidden_size', type=int, default=128, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512, help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1,help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: true]')
    parser.add_argument('--pmi_window_width', type=int, default=-1,help='Use PMI information for word node to word node')
    parser.add_argument('--sent_max_len', type=int, default=1300, help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--passage_length', type =int , default=10)
    parser.add_argument('--doc_max_timesteps', type=int, default=10, help='max length of documents (max timesteps of documents)')
    parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')
    parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
    parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')
    parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='for gradient clipping max gradient normalization')
    parser.add_argument('--mean_paragraphs', type=str, default=None, choices=[None, 'mean', 'mean_residual', 'mean_residual_add'],help='max length of documents (max timesteps of documents)')
    parser.add_argument('--revserse_metapath', action='store_true', default=False, help='Revserse the meta path of updating processing [default: False]')
    parser.add_argument('--retain_wp_relation', action='store_true', default=False, help='Revserse the meta path of updating processing [default: False]')
    parser.add_argument('--problem_type', type=str, default='classification', choices=['regression', 'classification', 'r2'], help='Regard problem as regression classification')
    parser.add_argument('--head', type=str, default='linear', choices=['linear', 'predictionhead'], help="Prediction Head")
    parser.add_argument('--pred_method', type=str, default=None, choices=[None, 'hec', 'ehg', 'acg', 'hsag', 'bert', 'sde', 'all_s', 'sde_s', 'hec_s', 'test_wdc', 'test_wdc2', 'test_wdc3'],help='max length of documents (max timesteps of documents)')

    # information embedding
    parser.add_argument('--cefr_word', action='store_true', default=False, help='Use CEFR vocabulary profile information')
    parser.add_argument('--cefr_info', type=str, default=None, choices=[None, 'embed_init', 'graph_init'], help="CEFR node embedding")
    parser.add_argument('--filled_pauses_word', action='store_true', default=False, help='Use disfluency tag information')
    parser.add_argument('--filled_pauses_info', type=str, default="embed_init", choices=['embed_init', 'graph_init'], help="Filled Pause embedding")

    # debug
    parser.add_argument('--final_attention', action='store_true', default=False, help='Use CEFR vocabulary profile information')

    # han in sentence to paragraph level feature
    parser.add_argument('--han_s', action='store_true', default=False, help='Use CEFR vocabulary profile information')

    # cheat baseline
    parser.add_argument('--baseline', action='store_true', default=False, help='Use ordinal entropy')

    # other experiments
    parser.add_argument('--language_use', action='store_true', default=False, help='Only train on Role-a-paly scenario')
    parser.add_argument('--memn2n', action='store_true', default=False, help='Only train on Role-a-paly scenario')

    # deep regression
    parser.add_argument('--bmse', action='store_true', default=False, help='use Balanced MSE')
    parser.add_argument('--imp', type=str, default=None, choices=[None, 'gai', 'bmc', 'bni'], help='implementation options')
    parser.add_argument('--gmm', type=str, default='gmm.pkl', help='path to preprocessed GMM')
    parser.add_argument('--init_noise_sigma', type=float, default=0.85, help='initial scale of the noise')
    parser.add_argument('--sigma_lr', type=float, default=1e-2, help='learning rate of the noise scale')
    parser.add_argument('--fix_noise_sigma', action='store_true', default=False, help='disable joint optimization')

    # Auxiliary objectives
    for obj in aux_objs:
        parser.add_argument('--use_{}'.format(obj), action='store_true', default=False,
                            help='Use {} objective during training.'.format(obj, obj))

    # ordinal entropy
    parser.add_argument('--oe', action='store_true', default=False, help='Use ordinal entropy')
    parser.add_argument('--oe_weight', type=float, default=1e-3, help='learning rate')
    
    # use document as features
    parser.add_argument('--use_doc', action='store_true', default=False, help='whether use doc or not')

    # Word CEFR as a new training objective
    parser.add_argument('--wcefr', action='store_true', default=False, help='add word cefr as training objective')
    parser.add_argument('--wcefr_reweight', action='store_true', default=False, help='Rewieght on word cefr loss')

    # Interviewer information
    parser.add_argument('--interviewer', action='store_true', default=False, help='Use interviewer information')

    parser.add_argument('--tsne', action='store_true', default=False, help='Save final hidden states form visuailizing in t-SNE.')
    
    
    parser.add_argument('--debug', action='store_true', default=False, help='Save final hidden states form visuailizing in t-SNE.')

    
    ## evaluation on train or validation set
    parser.add_argument('--which_set', default='eval', choices=['trn', 'dev', 'eval'], help='evaluation on train, validation or evaluation set')

    parser.add_argument('--save_gpu_mode', action='store_true', default=False, help='Save GPU when no enough vram')

    args = parser.parse_args()
    return args
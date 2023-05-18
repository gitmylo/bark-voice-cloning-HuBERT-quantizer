from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--path', required=True, help='The path containing your semantic tokens and wavs')
parser.add_argument('--mode', required=True, help='The mode to use', choices=['prepare', 'prepare2', 'train', 'test'])
parser.add_argument('--hubert-model', default='model/hubert/hubert_base_ls960.pt', help='The hubert model to use for preparing the data and later creation of semantic tokens.')
parser.add_argument('--train-save-epochs', default=1, type=int, help='The amount of epochs to train before saving')

args = parser.parse_args()

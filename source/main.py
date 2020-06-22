import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
import argparse

from codpackage.architecture import HRNet
from codpackage.datasampler.DataPreprocessor import DataPreprocessor
from codpackage.trainer.Trainer import Trainer

from configure.default import config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/home/xqwang/projects/camouflaged/dev/configure/w18.yaml'
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    parse_args()
    model = HRNet.get_model(config)
    preprocessor = DataPreprocessor(config)
    train_dataloader = preprocessor.get_train_dataloader()
    val_dataloader = preprocessor.get_val_dataloader()
    test_dataloader = None

    trainer = Trainer(model, train_dataloader, val_dataloader, config)
    trainer.train()

if __name__ == '__main__':
    main()
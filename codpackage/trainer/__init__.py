from . import SupervisedTrainer, ContrastiveTrainer, SSContrastiveTrainer

def get_trainer(config):
    return eval(config.TRAIN.TRAINER + '.' + config.TRAIN.TRAINER)

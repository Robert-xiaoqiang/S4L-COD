from . import SupervisedTrainer, ContrastiveTrainer

def get_trainer(config):
    return eval(config.TRAIN.TRAINER + '.' + config.TRAIN.TRAINER)

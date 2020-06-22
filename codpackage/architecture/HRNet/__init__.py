from .HRNet import HighResolutionNet

def get_model(cfg):
    model = HighResolutionNet(cfg)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
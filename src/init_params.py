"""Parameters utils"""

from mindspore.common.initializer import initializer, TruncatedNormal, XavierUniform


def init_net_param(network, initialize_mode='TruncatedNormal'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            if initialize_mode == 'TruncatedNormal':
                p.set_data(initializer(TruncatedNormal(), p.data.shape, p.data.dtype))
            elif initialize_mode == 'XavierUniform':
                p.set_data(initializer(XavierUniform(), p.data.shape, p.data.dtype))
            else:
                p.set_data(initialize_mode, p.data.shape, p.data.dtype)


def filter_checkpoint_parameter(param_dict):
    """remove useless parameters"""
    for key in list(param_dict.keys()):
        if 'multi_loc_layers' in key or 'multi_cls_layers' in key:
            del param_dict[key]

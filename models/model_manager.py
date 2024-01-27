from easyGOOD.utils.register import register
from easyGOOD.utils.config_reader import Union, CommonArgs, Munch
from easyGOOD.utils.initial import reset_random_seed


def load_model(name: str, config: Union[CommonArgs, Munch]) -> dir:
    return register.models[name](input_dim=config.dataset.dim_node, layer_num=config.model.model_layer, 
                          hidden=config.model.dim_hidden, output_dim=config.dataset.num_classes, activation=config.model.activation, dropout=config.model.dropout_rate,
                          use_bn=config.model.use_bn, last_activation=config.model.last_activation,
                          tau=config.model.tau, queue_size=config.model.queue_size, mm=config.train.mm, encoder_name=config.model.encoder_name,
                          num_clusters=config.model.num_clusters, prototypes_lr=config.model.prototypes_lr,
                          prototypes_iters=config.model.prototypes_iters, cmi_coefficient=config.model.cmi_coefficient,
                          dataset=config.dataset)
    
def load_sup_model(name: str, config):
    return register.models[name](config.dataset.dim_node, config.model.model_layer, config.model.dim_hidden, 
                config.dataset.num_classes, encoder_name=config.model.encoder_name, dropout=config.model.dropout_rate, config=config,
                use_bn=config.model.use_bn, last_activation=config.model.last_activation)
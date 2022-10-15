from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow_addons.optimizers import LAMB

OPTIMIZERS = {}
OPTIMIZERS["adam"] = lambda p: Adam(
    learning_rate=p.get("lr", 0.001),
    beta_1=p.get("beta_1", 0.9),
    beta_2=p.get("beta_2", 0.999),
    epsilon=p.get("epsilon", 1e-07),
    amsgrad=p.get("amsgrad", False),
)
OPTIMIZERS["lamb"] = lambda p: LAMB(
    learning_rate=p.get("lr", 0.001),
    beta_1=p.get("beta_1", 0.9),
    beta_2=p.get("beta_2", 0.999),
    epsilon=p.get("epsilon", 1e-06),
    weight_decay=p.get("weight_decay", 0.0),
    exclude_from_weight_decay=p.get("exclude_from_weight_decay", None),
    exclude_from_layer_adaptation=p.get("exclude_from_layer_adaptation", None),
)
OPTIMIZERS["rmsprop"] = lambda p: RMSprop(
    learning_rate=p.get("lr", 0.001),
    rho=p.get("rho", 0.9),
    momentum=p.get("momentum", 0.0),
    epsilon=p.get("epsilon", 1e-07),
    centered=p.get("centered", False),
)


def make_optimizer(params):
    opt_id = params.get("opt_id", "None")
    try:
        return OPTIMIZERS[opt_id](params)
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer name: {opt_id}") from exc

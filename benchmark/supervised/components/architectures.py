from tensorflow_similarity.architectures import (
    EfficientNetSim,
    ResNet18Sim,
    ResNet50Sim,
)

ARCHITECTURES = {}
ARCHITECTURES["effnet"] = lambda p: EfficientNetSim(
    input_shape=p["input_shape"],
    embedding_size=p.get("embedding", 128),
    variant=p.get("variant", "B0"),
    weights=p.get("weights", "imagenet"),
    trainable=p.get("trainable", "frozen"),
    l2_norm=p.get("l2_norm", True),
    include_top=p.get("include_top", True),
    pooling=p.get("pooling", "gem"),
    gem_p=p.get("gem_p", 3.0),
)
ARCHITECTURES["resnet50"] = lambda p: ResNet50Sim(
    input_shape=p["input_shape"],
    embedding_size=p.get("embedding", 128),
    weights=p.get("weights", "imagenet"),
    trainable=p.get("trainable", "frozen"),
    l2_norm=p.get("l2_norm", True),
    include_top=p.get("include_top", True),
    pooling=p.get("pooling", "gem"),
    gem_p=p.get("gem_p", 3.0),
)
ARCHITECTURES["resnet18"] = lambda p: ResNet18Sim(
    input_shape=p["input_shape"],
    embedding_size=p.get("embedding", 128),
    l2_norm=p.get("l2_norm", True),
    include_top=p.get("include_top", True),
    pooling=p.get("pooling", "gem"),
    gem_p=p.get("gem_p", 3.0),
)


def make_architecture(params):
    architecture_id = params.get("architecture_id", "None")
    try:
        return ARCHITECTURES[architecture_id](params)
    except KeyError as exc:
        raise ValueError(f"Unknown architecture name: {architecture_id}") from exc

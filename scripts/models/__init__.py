from models.graphsage import GraphSAGE
from models.typeprojection import TypeProjection
from models.linkpredictor import LinkPredictor

def load_model(config):
    if config["model"] == "graphsage":
        return GraphSAGE(
            config["in_channels"],
            config["hidden_channels"],
            config["out_channels"],
            config["dropout"])
    if config["model"] == "typeprojection":
        return TypeProjection(
            config["dim_types"],
            config["dim_output"])
    if config["model"] == "link_pred":
        return LinkPredictor(
            config["in_channels"],
            config["composition_function"]
            )
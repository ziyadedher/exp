from exp.models.conf_latent import ConfounderLatent
from exp.models.test import TensorflowTestModel, TorchTestModel
from ray.rllib.models import ModelCatalog

__all__ = ["TorchTestModel", "TensorflowTestModel", "ConfounderLatent"]

ModelCatalog.register_custom_model("TorchTestModel", TorchTestModel)
ModelCatalog.register_custom_model("TensorflowTestModel", TensorflowTestModel)
ModelCatalog.register_custom_model("ConfounderLatent", ConfounderLatent)

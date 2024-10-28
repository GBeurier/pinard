# config.py
import itertools
import dataclasses
from typing import List, Optional, Union

@dataclasses.dataclass
class Config:
    dataset: Union[str, object]
    x_pipeline: Optional[Union[str, object]] = None
    y_pipeline: Optional[Union[str, object]] = None
    model: Union[str, object] = None
    experiment: Optional[dict] = None
    seed: Optional[int] = None


# @dataclasses.dataclass
# class Configs_Generator:
#     datasets: List[str]
#     model_experiments: List[dict]  # List of tuples (model_config, experiment)
#     preparations: Optional[List[str]] = None
#     scalers: Optional[List[str]] = None
#     augmenters: Optional[List[str]] = None
#     preprocessings: Optional[List[str]] = None
#     reporting: Optional[dict] = None
#     seeds: Optional[List[int]] = None

#     def generate_configs(self):
#         self.preparations = self.preparations or [None]
#         self.scalers = self.scalers or [None]
#         self.preprocessings = self.preprocessings or [None]

#         for dataset, (model_config, experiment), preparation, scaler, preprocessing, seed in itertools.product(
#             self.datasets, self.model_experiments, self.preparations, self.scalers, self.preprocessings, self.seeds
#         ):
#             yield Config(dataset, model_config, preparation, scaler, preprocessing, experiment, seed, self.reporting)

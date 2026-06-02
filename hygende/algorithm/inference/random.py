from abc import ABC, abstractmethod
import os
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pulp
import random
import re

from . import inference_register
from .base import Inference
from ..summary_information import SummaryInformation
from ...prompt import BasePrompt
from ...tasks import BaseTask


@inference_register.register("random")
class RandomInference(Inference):
    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        train_data: pd.DataFrame,
        task: BaseTask,
    ):
        super().__init__(api, prompt_class, train_data, task)

    def batched_predict(
        self,
        data: pd.DataFrame,
        idx_hyp_pair=List[Tuple[int, Dict[str, SummaryInformation]]],
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs,
    ):
        """
        Makes a batch of predictions on hypotheses.

        Parameters:
            data: the data to predict on
            idx_hyp_pair: a list of tuples of indices and hypothesis banks
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        # Use the prompt class to create the batch of prompts
        prompt_inputs = [
            self.prompt_class.inference(hyp_bank, data, index)
            for index, hyp_bank in idx_hyp_pair
        ]
        
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        
        predictions = [self.task.extract_label(response) for response in responses]
        actual_labels = [data["label"][index] for index, _ in idx_hyp_pair]

        return predictions, actual_labels

    def run_inference_final(
        self,
        data,
        hyp_bank,
        adaptive_num_hypotheses=5,
        cache_seed=None,
        max_concurrent=3,
        generate_kwargs={},
        **kwargs,
    ):
        """
        Function for testing with randomly selected hypotheses

        Parameters:
            data: the data to predict on
            hyp_bank: the hypotheses that we want to predict from
            adaptive_num_hypotheses: number of hypotheses to randomly select
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests
        """
        num_samples = len(data)
        all_hypotheses = list(hyp_bank.keys())
        
        # Ensure the number of hypotheses to select does not exceed the total number of hypotheses
        k = min(adaptive_num_hypotheses, len(all_hypotheses))
        
        # Prepare the list of index-hypothesis pairs
        idx_hyp_pair = []
        for i in range(num_samples):
            # Randomly select k hypotheses
            selected_hypotheses = random.sample(all_hypotheses, k) if len(all_hypotheses) >= k else all_hypotheses
            
            # Create hypothesis dictionary
            selected_hyp_bank = {hyp: hyp_bank[hyp] for hyp in selected_hypotheses}
            idx_hyp_pair.append((i, selected_hyp_bank))

        # Run batched prediction
        return self.batched_predict(
            data,
            idx_hyp_pair,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
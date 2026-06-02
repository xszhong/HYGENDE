import argparse
import logging
import re
import time
import pickle
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import json

import random
from typing import Callable, Tuple, Union
import torch
import numpy as np

from hygende.extract_label import extract_label_register

from hygende.tasks import BaseTask
from hygende.prompt import BasePrompt
from hygende.utils import set_seed


from hygende.algorithm.summary_information import SummaryInformation
from hygende.LLM_wrapper import (
    GPTWrapper,
    LLMWrapper,
    LocalVllmWrapper,
    llm_wrapper_register,
)

from hygende.algorithm.generation import DefaultGeneration
from hygende.algorithm.inference import (
    WeightBasedInference,
    RelevanceBasedInference,
    RandomInference,
)
from hygende.algorithm.replace import DefaultReplace
from hygende.algorithm.update import SamplingUpdate, DefaultUpdate
from hygende.logger_config import LoggerConfig

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("HyGende")


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def main():
    # set up tools
    start_time = time.time()

    # For detailed argument descriptions, please see `hygende_cmd/generation.py`
    task_config_path = "./data/paddle/config.yaml"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_path = "./Meta-Llama-3-8B-Instruct"
    model_type = "vllm"

    max_num_hypotheses = 20
    
    old_hypothesis_file = None
    num_init = 10
    num_train = 600
    num_test = 300
    num_val = 100
    k = 10
    alpha = 5e-1
    update_batch_size = 10
    num_hypotheses_to_update = 1
    save_every_10_examples = 10
    init_batch_size = 10
    init_hypotheses_per_batch = 10
    cache_seed = None
    temperature = 1e-5
    max_tokens = 1000
    seeds = [42]

    output_folder = f"./outputs/paddle/gen_{model_name}_train_{num_train}_hypothesis_{max_num_hypotheses}/"

    os.makedirs(output_folder, exist_ok=True)
    api = llm_wrapper_register.build(model_type)(model=model_name, path_name=model_path)

    # If implementing a new task, you need to create a new extract_label function and pass in the Task constructor.
    # For existing tasks, you can use the extract_label_register.
    task = BaseTask(
        task_config_path, extract_label=None, from_register=extract_label_register
    )

    for seed in seeds:
        set_seed(seed)
        train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
        prompt_class = BasePrompt(task)
        inference_class = WeightBasedInference(api, prompt_class, train_data, task)
        generation_class = DefaultGeneration(api, prompt_class, inference_class, task)

        update_class = DefaultUpdate(
            generation_class=generation_class,
            inference_class=inference_class,
            replace_class=DefaultReplace(max_num_hypotheses),
            save_path=output_folder,
            num_init=num_init,
            k=k,
            alpha=alpha,
            update_batch_size=update_batch_size,
            num_hypotheses_to_update=num_hypotheses_to_update,
            save_every_n_examples=save_every_10_examples,
        )

        hypotheses_bank = {}
        if old_hypothesis_file is None:
            hypotheses_bank = update_class.batched_initialize_hypotheses(
                num_init,
                init_batch_size=init_batch_size,
                init_hypotheses_per_batch=init_hypotheses_per_batch,
                cache_seed=cache_seed,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            update_class.save_to_json(
                hypotheses_bank,
                sample=num_init,
                seed=seed,
                epoch=0,
            )
        else:
            dict = load_dict(old_hypothesis_file)
            for hypothesis in dict:
                hypotheses_bank[hypothesis] = SummaryInformation.from_dict(
                    dict[hypothesis]
                )
        for epoch in range(1):
            hypotheses_bank = update_class.update(
                current_epoch=epoch,
                hypotheses_bank=hypotheses_bank,
                current_seed=seed,
                cache_seed=cache_seed,
            )
            update_class.save_to_json(
                hypotheses_bank,
                sample="final",
                seed=seed,
                epoch=epoch,
            )

    # print experiment info
    logger.info(f"Total time: {time.time() - start_time} seconds")
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     logger.info(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()

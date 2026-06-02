import argparse
import re
import time
import pickle
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import json
import logging

import random
from typing import Union
import torch
import numpy as np

from hygende.extract_label import depression_extract_label, bidetweet_extract_label, fourdetweet_extract_label
u
from hygende.tasks import BaseTask
from hygende.prompt import BasePrompt
from hygende.utils import (
    get_results,
    set_seed,
)

from hygende.algorithm.summary_information import SummaryInformation
from hygende.LLM_wrapper import (
    GPTWrapper,
    LLMWrapper,
    LocalVllmWrapper,
    llm_wrapper_register,
)

from hygende.algorithm.inference import (
    WeightBasedInference,
    RelevanceBasedInference,
    RandomInference,
    inference_register
)
from hygende.logger_config import LoggerConfig

logger = LoggerConfig.get_logger("HyGende")
LoggerConfig.setup_logger(
        logging.DEBUG,
)

def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main():
    start_time = time.time()

    # For detailed argument descriptions, please see `hygende_cmd/inference.py`
    task_config_path = "./data/paddle/config.yaml"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_path = "./Meta-Llama-3-8B-Instruct"
    model_type = "vllm"

    inference_type = "weight"

    max_num_hypotheses = 20
    adaptive_num_hypotheses = 5
    num_train = 200
    num_test = 300
    num_val = 100
    use_valid = False
    seeds = [42]
    cache_seed = None
    max_concurrent = 3
    temperature = 1e-5
    max_tokens = 1000

    hypothesis_file = f"./outputs/paddle/gen_{model_name}_train_{num_train}_hypothesis_{max_num_hypotheses}/hypotheses_training_sample_final_seed_42_epoch_0.json"

    accuracy_all = []
    
    dict = load_dict(hypothesis_file)
    hyp_bank = {}

    task = BaseTask(task_config_path, extract_label=depression_extract_label)

    for hypothesis in dict:
        hyp_bank[hypothesis] = SummaryInformation.from_dict(dict[hypothesis])

    assert adaptive_num_hypotheses <= len(
        hyp_bank
    ), f"The number of hypotheses chosen in relevance-based inference must be less than the total number of hypotheses"

    api = llm_wrapper_register.build(model_type)(model=model_name, path_name=model_path)

    for seed in seeds:
        set_seed(seed)
        train_data, test_data, val_data = task.get_data(
            num_train, num_test, num_val, seed
        )
        prompt_class = BasePrompt(task)
        inference_class = inference_register.build(inference_type)(api, prompt_class, train_data, task)

        if use_valid:
            logger.info("Using validation data")
            test_data = val_data
        else:
            logger.info("Using test data")

        pred_list, label_list = inference_class.run_inference_final(
            test_data,
            hyp_bank,
            adaptive_num_hypotheses=adaptive_num_hypotheses,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            generate_kwargs={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        results_dict = get_results(pred_list, label_list)

        logger.info(f"Accuracy for seed {seed}: {results_dict['accuracy']}")

        # print the wrong indices
        wrong_indices = [
            i for i in range(len(pred_list)) if pred_list[i] != label_list[i]
        ]
        logger.info(f"Wrong indices: {wrong_indices}")
        accuracy_all.append(results_dict["accuracy"])

    logger.info(f"Averaged accuracy: {sum(accuracy_all)/len(accuracy_all):.3f}")

    # print experiment info
    logger.info(f"Total time: {time.time() - start_time} seconds")
    # if api.model in GPT_MODELS:
    #     logger.info(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()

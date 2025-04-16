import os
import time
import json
import multiprocessing
import http.client

from typing import Collection, Any
from implementation_cl import sampler

from prompts.prompts import *


api_key = 'Bearer sk-OmRJlpj2aI4A3GLvA4Bd841fCfB04b3e9eF6D0D9984f1719'

server_url = 'api.bltcy.ai'
max_tokens = 1024
model_name = 'chatgpt-4o-latest'

spec_file = 'flowshop_spec_priority.py'
add_prompt = base_prompt

max_attempts = 5
max_sample_num_per_stage = 10

def _eval_metric(baseline: dict, result: dict) -> bool:
    baseline_avg = sum(baseline.values()) / len(baseline)
    result_avg = sum(result.values()) / len(result)
    return result_avg >= baseline_avg


def _trim_preface_of_body(sample: str) -> str:
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False

    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break

    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno+1:]:
            code += line + '\n'
        return code

    return sample


class LLMAPI(sampler.LLM):
    def __init__(
            self,
            samples_per_prompt: int,
            trim=True
    ):
        super().__init__(samples_per_prompt)
        self._additional_prompt = add_prompt
        self._trim = trim

    def draw_samples(
            self,
            prompt: str
    ) -> Collection[str]:
        return [
            self._draw_sample(prompt)
            for _ in range(self._samples_per_prompt)
        ]

    def _draw_sample(
            self,
            content: str
    ) -> str:
        prompt = '\n'.join([content, self._additional_prompt])

        while True:
            try:
                conn = http.client.HTTPSConnection(server_url)
                payload = json.dumps({
                    'max_tokens': max_tokens,
                    'model': model_name,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                })
                headers = {
                    'Authorization': api_key,
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }

                conn.request('POST', '/v1/chat/completions',
                             payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                response = data['choices'][0]['message']['content']

                if self._trim:
                    response = _trim_preface_of_body(response)
                return response

            except Exception:
                time.sleep(2)
                continue


from implementation_cl import evaluator
from implementation_cl import evaluator_accelerate


class Sandbox(evaluator.Sandbox):
    def __init__(
            self,
            verbose=False,
            numba_accelerate=False
    ):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program:            str,
            function_to_run:    str,
            function_to_evolve: str,
            inputs:             Any,
            test_input:         str,
            timeout_seconds:    int,
            **kwargs
    ) -> tuple[Any, bool]:
        dataset = {
            test_input: inputs[test_input]
        }

        try:
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target = self._compile_and_run_function,
                args = (
                    program,
                    function_to_run,
                    function_to_evolve,
                    dataset,
                    self._numba_accelerate,
                    result_queue
                )
            )
            process.start()
            process.join(timeout=timeout_seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                results = None, False
            else:
                if not result_queue.empty():
                    results = result_queue.get_nowait()
                else:
                    results = None, False

            return results
        except:
            return None, False

    def _compile_and_run_function(
            self,
            program,
            function_to_run,
            function_to_evolve,
            dataset,
            numba_accelerate,
            result_queue
    ):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program = program,
                    function_to_evolve = function_to_evolve
                )

            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)

            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return

            result_queue.put((results, True))

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"[Sandbox Error]: {error_msg}")
            result_queue.put((None, False))


from flowshop_test.utils import load_datasets

from implementation_cl import funsearch
from implementation_cl import config


if __name__ == '__main__':
    data_path = '/Users/cuiguangyuan/Documents/CityU/SemesterB/Artificial Intelligence/project/Funsearch_on_flowshop/data'
    datasets = {}

    for subfolder in ['carlier', 'heller', 'reeves']:
        datasets.update(load_datasets(os.path.join(data_path, subfolder)))

    print(f'Successfully loaded {len(datasets)} datasets.')

    """Instance with different stages"""
    instances = [
        # Stage 1
        {
            'carlier1.txt': datasets['carlier1.txt'],  # 11 * 5
            'carlier2.txt': datasets['carlier2.txt'],  # 13 * 4
            'carlier3.txt': datasets['carlier3.txt'],  # 12 * 5
            'carlier4.txt': datasets['carlier4.txt']   # 14 * 4
        },
        # Stage 2
        {
            'heller2.txt': datasets['heller2.txt'],    # 20 * 10
            'reeves13.txt': datasets['reeves13.txt'],  # 30 * 15
            'reeves14.txt': datasets['reeves14.txt'],  # 30 * 15
            'reeves15.txt': datasets['reeves15.txt']   # 30 * 15
        },
        # Stage 3
        {
            'reeves16.txt': datasets['reeves16.txt'],  # 50 * 10
            'reeves17.txt': datasets['reeves17.txt'],  # 50 * 10
            'reeves18.txt': datasets['reeves18.txt']   # 50 * 10
        }
    ]

    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=30)

    funsearch.main(
        spec_file_path=spec_file,
        inputs=instances,
        config=config,
        max_sample_nums_per_stage=max_sample_num_per_stage,
        max_attempts_per_stage=max_attempts,
        class_config=class_config,
        verbose=True,
        log_dir=f'./logs_cl/evaluator_log/{spec_file.split('.')[0]}'
    )

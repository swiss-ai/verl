# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
import pebble
import asyncio

from .logging_utils import get_reward_logger, log_reward_error

logger = get_reward_logger(__name__)

_pool = None
_pool_lock = threading.Lock()

def _pool_init():
    import resource, sys
    lim = 8 << 30
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    sys.set_int_max_str_digits(50_000)

def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(
                    max_workers=4, 
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_pool_init,
                    max_tasks_per_child=200
                )
    return _pool


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    # TODO: Find a better dependency isolation strategy
    # This patch forces subprocesses to load math_verify
    # from the shared path instead of whatever Ray/container imports first
    math_verify_pythonpath = os.environ.get("MATH_VERIFY_PYTHONPATH")
    if math_verify_pythonpath:
        for path in reversed(math_verify_pythonpath.split(os.pathsep)):
            if path and path not in sys.path:
                sys.path.insert(0, path)
        for module_name in list(sys.modules):
            if (
                module_name == "antlr4"
                or module_name.startswith("antlr4.")
                or module_name == "math_verify"
                or module_name.startswith("math_verify.")
                or module_name == "latex2sympy2_extended"
                or module_name.startswith("latex2sympy2_extended.")
            ):
                del sys.modules[module_name]

    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(
            1.0 if any(verify(g, p) for g in extracted_gold) else 0.0
            for p in extracted_pred
        )
    return 0.0


def compute_score(
    model_output: str,
    ground_truth: str,
    timeout_score: float = 0,
    timeout: float = 30.0,
    data_source: str | None = None,
) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(
            _verify_in_subprocess, ground_truth_boxed, model_output
        )
        ret_score = future.result(timeout=timeout)
    except FuturesTimeoutError:
        ret_score = timeout_score
    except Exception as exc:
        log_reward_error(
            logger,
            "math_verify",
            "returning 0 reward",
            data_source=data_source,
            exc=exc,
        )
    return ret_score

_fast_pool = None
_slow_pool = None
_fast_pool_lock = threading.Lock()
_slow_pool_lock = threading.Lock()

def _forkserver_context():
    # forkserver makes the post-kill worker respawn cheap (~ms fork from a
    # preloaded process instead of spawn + fresh sympy import), which is what
    # keeps the fast-tier eviction affordable.
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(["math_verify.grader", "math_verify.parser"])
    return context

def _get_fast_pool():
    global _fast_pool
    if _fast_pool is None:
        with _fast_pool_lock:
            if _fast_pool is None:
                _fast_pool = pebble.ProcessPool(
                    max_workers=8,
                    max_tasks=0,
                    initializer=_pool_init,
                    context=_forkserver_context()
                )
    return _fast_pool

def _get_slow_pool():
    global _slow_pool
    if _slow_pool is None:
        with _slow_pool_lock:
            if _slow_pool is None:
                _slow_pool = pebble.ProcessPool(
                    max_workers=4,
                    max_tasks=0,
                    initializer=_pool_init,
                    context=_forkserver_context()
                )
    return _slow_pool

async def compute_score_async(
    model_output: str,
    ground_truth: str,
    timeout_score: float = 0.0,
    fast_timeout = 3.0,
    timeout: float = 15.0,
    data_source: str | None = None,
) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    future = _get_fast_pool().schedule(
        _verify_in_subprocess,
        args=[ground_truth_boxed, model_output],
        timeout=fast_timeout
    )
    retry = False
    try:
        ret_score = await asyncio.wrap_future(future)
    except (TimeoutError, pebble.ProcessExpired):
        retry = True
    except Exception as exc:
        log_reward_error(
            logger,
            "math_verify",
            "returning 0 reward",
            data_source=data_source,
            exc=exc,
        )
    if not retry:
        return ret_score
    slow_future = _get_slow_pool().schedule(
        _verify_in_subprocess,
        args=[ground_truth_boxed, model_output],
        timeout=timeout
    )
    try:
        ret_score = await asyncio.wrap_future(slow_future)
    except (TimeoutError, pebble.ProcessExpired):
        return timeout_score
    except Exception as exc:
        log_reward_error(
            logger,
            "math_verify",
            "returning 0 reward",
            data_source=data_source,
            exc=exc,
        )
    return ret_score

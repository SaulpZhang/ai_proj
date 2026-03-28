# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import datetime
import time
from typing import Any

from absl import logging
import numpy as np
import scipy

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  """
  code: str
  version_generated: int
  island_id: int


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: config_lib.ProgramsDatabaseConfig,
      template: code_manipulation.Program,
      function_to_evolve: str,
  ) -> None:
    self._config: config_lib.ProgramsDatabaseConfig = config
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve

    # Initialize empty islands.
    self._islands: list[Island] = []
    for _ in range(config.num_islands):
      self._islands.append(
          Island(template, function_to_evolve, config.functions_per_prompt,
                 config.cluster_sampling_temperature_init,
                 config.cluster_sampling_temperature_period))
    self._best_score_per_island: list[float] = (
        [-float('inf')] * config.num_islands)
    self._best_program_per_island: list[code_manipulation.Function | None] = (
        [None] * config.num_islands)
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        [None] * config.num_islands)
    self._scores_log_path = config.scores_log_path
    self._best_programs_log_path = config.best_programs_log_path
    self._run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    self._logged_global_best_score: float | None = None

    self._initialize_scores_log()
    self._initialize_best_programs_log(function_to_evolve)

    self._last_reset_time: float = time.time()

  def _initialize_scores_log(self) -> None:
    """Creates score log header if file does not exist or is empty."""
    try:
      with open(self._scores_log_path, 'x', encoding='utf-8') as f:
        f.write('run_id,timestamp,island_id,from_score,to_score,global_best\n')
    except FileExistsError:
      with open(self._scores_log_path, 'r', encoding='utf-8') as f:
        has_content = bool(f.read(1))
      if not has_content:
        with open(self._scores_log_path, 'a', encoding='utf-8') as f:
          f.write('run_id,timestamp,island_id,from_score,to_score,global_best\n')

  def _initialize_best_programs_log(self, function_to_evolve: str) -> None:
    """Appends run-start metadata to the best-programs log file."""
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    with open(self._best_programs_log_path, 'a', encoding='utf-8') as f:
      f.write('\n')
      f.write(f'=== RUN_START run_id={self._run_id} timestamp={timestamp} ===\n')
      f.write(f'function_to_evolve={function_to_evolve} num_islands={self._config.num_islands}\n')

  def _format_score_for_csv(self, score: float | None) -> str:
    """Formats finite scores as numbers and non-finite as empty fields."""
    if score is None:
      return ''
    if np.isfinite(score):
      return str(score)
    return ''

  def _append_score_change_log(
      self,
      island_id: int,
      previous_score: float,
      new_score: float,
  ) -> None:
    """Appends one score-change record to the CSV-style score log."""
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    global_best = self.get_global_best_score()
    with open(self._scores_log_path, 'a', encoding='utf-8') as f:
      f.write(
          f'{self._run_id},{timestamp},{island_id},'
          f'{self._format_score_for_csv(previous_score)},'
          f'{self._format_score_for_csv(new_score)},'
          f'{self._format_score_for_csv(global_best)}\n')

  def _append_best_program_log(
      self,
      tag: str,
      island_id: int,
      score: float,
      scores_per_test: ScoresPerTest | None,
      program: code_manipulation.Function | None,
  ) -> None:
    """Appends island/global best program details to dedicated txt log."""
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    with open(self._best_programs_log_path, 'a', encoding='utf-8') as f:
      scores_text = str(scores_per_test) if scores_per_test is not None else '{}'
      f.write(
          f'[{timestamp}] {tag} run_id={self._run_id} island={island_id} '
          f'score={score} scores_per_test={scores_text}\n')
      f.write('PROGRAM_START\n')
      f.write(str(program) if program is not None else 'None')
      f.write('\nPROGRAM_END\n')

  def _log_global_best_if_improved(self) -> None:
    """Logs global best program snapshot when score is improved."""
    best_score = self.get_global_best_score()
    if best_score is None:
      return
    if (self._logged_global_best_score is not None
        and best_score <= self._logged_global_best_score):
      return

    best_island_id = int(np.argmax(self._best_score_per_island))
    self._append_best_program_log(
        tag='GLOBAL_BEST_UPDATE',
        island_id=best_island_id,
        score=best_score,
        scores_per_test=self._best_scores_per_test_per_island[best_island_id],
        program=self._best_program_per_island[best_island_id],
    )
    self._logged_global_best_score = best_score

  def get_prompt(self) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    island_id = np.random.randint(len(self._islands))
    code, version_generated = self._islands[island_id].get_prompt()
    return Prompt(code, version_generated, island_id)

  def get_global_best_score(self) -> float | None:
    """Returns best score across all islands, or `None` if unavailable."""
    if not self._best_score_per_island:
      return None
    best = float(np.max(self._best_score_per_island))
    if not np.isfinite(best):
      return None
    return best

  def _register_program_in_island(
      self,
      program: code_manipulation.Function,
      island_id: int,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the specified island."""
    previous_score = self._best_score_per_island[island_id]
    self._islands[island_id].register_program(program, scores_per_test)
    score = _reduce_score(scores_per_test)
    if score > self._best_score_per_island[island_id]:
      self._best_program_per_island[island_id] = program
      self._best_scores_per_test_per_island[island_id] = scores_per_test
      self._best_score_per_island[island_id] = score
      logging.info('Best score of island %d increased to %s', island_id, score)
      self._append_score_change_log(island_id, previous_score, score)
      self._append_best_program_log(
          tag='ISLAND_BEST_UPDATE',
          island_id=island_id,
          score=score,
          scores_per_test=scores_per_test,
          program=program,
      )
      self._log_global_best_if_improved()

  def register_program(
      self,
      program: code_manipulation.Function,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    if island_id is None:
      # This is a program added at the beginning, so adding it to all islands.
      for island_id in range(len(self._islands)):
        self._register_program_in_island(program, island_id, scores_per_test)
    else:
      self._register_program_in_island(program, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    if (time.time() - self._last_reset_time > self._config.reset_period):
      self._last_reset_time = time.time()
      self.reset_islands()

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
          self._template,
          self._function_to_evolve,
          self._config.functions_per_prompt,
          self._config.cluster_sampling_temperature_init,
          self._config.cluster_sampling_temperature_period)
      self._best_score_per_island[island_id] = -float('inf')
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_program_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      self._register_program_in_island(founder, island_id, founder_scores)


class Island:
  """A sub-population of the programs database."""

  def __init__(
      self,
      template: code_manipulation.Program,
      function_to_evolve: str,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self._cluster_sampling_temperature_period = (
        cluster_sampling_temperature_period)

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs: int = 0

  def register_program(
      self,
      program: code_manipulation.Function,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Stores a program on this island, in its appropriate cluster."""
    signature = _get_signature(scores_per_test)
    if signature not in self._clusters:
      score = _reduce_score(scores_per_test)
      self._clusters[signature] = Cluster(score, program)
    else:
      self._clusters[signature].register_program(program)
    self._num_programs += 1

  def get_prompt(self) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    signatures = list(self._clusters.keys())
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])

    # Convert scores to probabilities using softmax with temperature schedule.
    period = self._cluster_sampling_temperature_period
    temperature = self._cluster_sampling_temperature_init * (
        1 - (self._num_programs % period) / period)
    probabilities = _softmax(cluster_scores, temperature)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

    idx = np.random.choice(
        len(signatures), size=functions_per_prompt, p=probabilities)
    chosen_signatures = [signatures[i] for i in idx]
    implementations = []
    scores = []
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      implementations.append(cluster.sample_program())
      scores.append(cluster.score)

    indices = np.argsort(scores)
    sorted_implementations = [implementations[i] for i in indices]
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated

  def _generate_prompt(
      self,
      implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    implementations = copy.deepcopy(implementations)  # We will mutate these.

    # Format the names and docstrings of functions to be included in the prompt.
    versioned_functions: list[code_manipulation.Function] = []
    for i, implementation in enumerate(implementations):
      new_function_name = f'{self._function_to_evolve}_v{i}'
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.
      if i >= 1:
        implementation.docstring = (
            f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
      # If the function is recursive, replace calls to itself with its new name.
      implementation = code_manipulation.rename_function_calls(
          str(implementation), self._function_to_evolve, new_function_name)
      versioned_functions.append(
          code_manipulation.text_to_function(implementation))

    # Create the header of the function to be generated by the LLM.
    next_version = len(implementations)
    new_function_name = f'{self._function_to_evolve}_v{next_version}'
    header = dataclasses.replace(
        implementations[-1],
        name=new_function_name,
        body='',
        docstring=('Improved version of '
                   f'`{self._function_to_evolve}_v{next_version - 1}`.'),
    )
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    return str(prompt)


class Cluster:
  """A cluster of programs on the same island and with the same Signature."""

  def __init__(self, score: float, implementation: code_manipulation.Function):
    self._score = score
    self._programs: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))]

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    self._programs.append(program)
    self._lengths.append(len(str(program)))

  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorther programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    probabilities = _softmax(-normalized_lengths, temperature=1.0)
    return np.random.choice(self._programs, p=probabilities)

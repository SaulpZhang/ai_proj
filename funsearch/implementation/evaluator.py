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

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
import math
import signal
from typing import Any

from funsearch.implementation import code_manipulation
from funsearch.implementation import programs_database

import json

import logging

logger = logging.getLogger(__name__)


def _strip_markdown_code_fences(code: str) -> str:
  """Removes markdown code-fence lines from generated snippets."""
  cleaned_lines = []
  for line in code.splitlines():
    if line.strip().startswith('```'):
      continue
    cleaned_lines.append(line)
  return '\n'.join(cleaned_lines).strip('\n')


def _normalize_body_indentation(body: str) -> str:
  """Normalizes function-body indentation to a 2-space baseline."""
  lines = body.splitlines()
  non_empty_lines = [line for line in lines if line.strip()]
  if not non_empty_lines:
    return '\n\n'

  min_indent = min(len(line) - len(line.lstrip(' ')) for line in non_empty_lines)
  normalized_lines = []
  for line in lines:
    if line.strip():
      dedented = line[min_indent:] if len(line) >= min_indent else line.lstrip(' ')
      normalized_lines.append(f'  {dedented}')
    else:
      normalized_lines.append('')
  return '\n'.join(normalized_lines) + '\n\n'


def _extract_body_from_top_level_function(code: str) -> str | None:
  """Returns body of the first top-level function in `code`, if any."""
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return None

  for node in tree.body:
    if isinstance(node, ast.FunctionDef):
      if not node.body:
        return ''
      lines = code.splitlines()
      body_start_line = node.body[0].lineno - 1
      body_lines = lines[body_start_line:node.end_lineno]
      return _normalize_body_indentation('\n'.join(body_lines))
  return None


def _indent_code_block(code: str) -> str:
  """Indents every non-empty line by 2 spaces."""
  indented_lines = []
  for line in code.splitlines():
    if line.strip():
      indented_lines.append(f'  {line}')
    else:
      indented_lines.append(line)
  return '\n'.join(indented_lines)


def _trim_wrapped_body(code_body: str) -> str:
  """Tries to parse body by wrapping it into a fake function."""
  code = f'def fake_function_header():\n{code_body}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None and code:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      if e.lineno is None or e.lineno <= 1:
        code = ''
        break
      next_code = '\n'.join(code.splitlines()[:e.lineno - 1])
      if next_code == code:
        code = ''
        break
      code = next_code
  if not code:
    # Nothing could be saved from `generated_code`
    return ''
  assert tree is not None

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return _normalize_body_indentation('\n'.join(body_lines))

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''

  cleaned_code = _strip_markdown_code_fences(generated_code)
  if not cleaned_code:
    return ''

  # Some models return a complete function definition rather than only the
  # function body. Handle that case directly.
  function_body = _extract_body_from_top_level_function(cleaned_code)
  if function_body is not None:
    return function_body

  trimmed = _trim_wrapped_body(cleaned_code)
  if trimmed:
    return trimmed

  # If model output is valid code body but not indented yet, try once with
  # normalization for function-body indentation.
  return _trim_wrapped_body(_indent_code_block(cleaned_code))


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  body = _trim_function_body(generated_code)
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  return evolved_function, str(program)


class Sandbox:
  """Sandbox for executing generated code."""

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: str,
      timeout_seconds: int,
  ) -> tuple[Any, bool, str | None]:
    """Returns result, whether execution succeeded and failure type."""
    namespace: dict[str, Any] = {}

    class _TimeoutError(Exception):
      pass

    def _timeout_handler(signum: int, frame: Any) -> None:
      del signum, frame
      raise _TimeoutError()

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
      signal.signal(signal.SIGALRM, _timeout_handler)
      signal.alarm(timeout_seconds)

      exec(program, namespace)  # pylint: disable=exec-used
      function = namespace.get(function_to_run)
      if not callable(function):
        logger.error(
            'Sandbox failure: function `%s` is not callable after exec.',
            function_to_run)
        return None, False, 'FunctionNotCallable'

      result = function(test_input)
      return result, True, None
    except Exception as e:
      logger.exception(
          'Sandbox exception while running `%s` on input type `%s`.',
          function_to_run,
          type(test_input).__name__)
      return None, False, type(e).__name__
    finally:
      signal.alarm(0)
      signal.signal(signal.SIGALRM, previous_handler)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      timeout_seconds: int = 30,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox()

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)
    if not new_function.body.strip():
      logger.info('Discarded sample: empty evolved function body after trim.')
      return

    calls_ancestor = _calls_ancestor(program, self._function_to_evolve)
    if calls_ancestor:
      logger.info('Discarded sample: generated code calls an ancestor function.')
      return

    scores_per_test = {}
    for current_input in self._inputs:
      logger.info("Starting run the code")
      test_output, runs_ok, error_type = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      if runs_ok and test_output is not None:
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        if not math.isfinite(float(test_output)):
          logger.info(
              'Run failed: function returned non-finite score %s.', test_output)
          logger.info("Finished running the code")
          continue
        logger.info("Code ran successfully, score: %s", test_output)
        scores_per_test[json.dumps(current_input)] = test_output
      elif not runs_ok:
        logger.info(
            'Run failed: sandbox execution returned runs_ok=False, error_type=%s.',
            error_type or 'UnknownError')
      else:
        logger.info('Run failed: function returned None.')
      logger.info("Finished running the code")
    if scores_per_test:
      self._database.register_program(new_function, island_id, scores_per_test)
    else:
      logger.info('Discarded sample: no valid scores were produced for any input.')

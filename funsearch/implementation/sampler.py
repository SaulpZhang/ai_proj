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

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database
from funsearch.implementation import code_manipulation

import openai

class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.client = openai.OpenAI(base_url="https://api.bltcy.ai/v1",api_key="sk-KcOna1MllF2BWDRIeXpcHAY92TKvHox3KnfpXsBZHiydDD0n",timeout=120,)

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""

    system_prompt = "你是一位顶级算法与Python编程专家，专注于代码迭代优化。\
    核心规则：\
    1. 严格遵循用户提供的代码格式，仅生成新版本的目标函数，不修改原有代码结构；\
    2. 必须输出完整可运行的函数体、明确的return返回值，保证代码语法100%正确；\
    3. 仅在核心逻辑处添加极简注释，禁止冗余说明、无关文本和markdown格式；\
    4. 保证生成的函数与原有版本的入参、出参格式完全一致，仅优化算法逻辑; \
    5. 必须返回可直接运行的代码，而不是将代码作为注释输出;\
    6. ```python和```等markdown格式必须删除，禁止输出任何markdown格式。"

    user_content = f"下方代码中，方法名后缀数字越大代表版本越新。请基于现有代码，实现更高版本的新方法，严格遵循上面的规则。尽量不要生成功能相似的代码，而是要在算法逻辑上进行创新和优化。代码如下：{prompt}"

    llm_response = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_content}],
        max_tokens=2048,
        temperature=0.7,
    )
    return llm_response.choices[0].message.content # type: ignore

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    while True:
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code)
      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        sample = self.remove_note(sample)
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
  
  def remove_note(self, sample: str) -> str:
    try:
      program = code_manipulation.text_to_program(sample)
      return program.get_function(program.functions[0].name).body
    except Exception:
      # If the generated code cannot be parsed, we return it as is, and let the
      # evaluator handle the syntax error.
      return sample
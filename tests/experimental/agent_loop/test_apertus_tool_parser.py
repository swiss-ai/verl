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
import pytest
from transformers import AutoTokenizer
from verl.experimental.agent_loop.tool_parser import Apertus2509ToolParser


@pytest.mark.asyncio
async def test_gpt_oss_tool_parser():
    sample_response = """
I'll help you find rhyming words for \"scrawny\" categorized by syllable count. Let me find the rhymes for this word.<|tools_prefix|>[{\"find_rhymes\": {\"input_word\": \"scrawny\"}}]<|tools_suffix|><|tool_output_start|>\"error: Tool 'find_rhymes' on 'forceconstant_lyrical_mcp' timed out after 30.0s\"<|tool_output_end|>It seems there was a timeout when trying to retrieve rhymes. Let me try to provide a practical response based on the single-syllable keyword \"scrawny\" to see if we get rhyming suggestions.\n\nFor words that rhyme with \"scrawny\" (category: 1 syllable):\n\nHere are some 1-syllable words that rhyme with or are close to \"scrawny\":\n- Mawny\n- Mawny\n- Prowly\n- Snooty\n- Laughton \n- Laughtrey\n- Roughty\n- Chawner \n- Jauney  \n- Naughty\n- Clawny\n- Shawnee\n- Hawn\n- Brawn  \n- Drawn\n- Farren\n\nAlso note that \"scrawny\" itself is a negative descriptor meaning thin or bony, and partial words like \"scaw\" and \"awn\" might also function similarly conceptually.\n\nPlease let me know if you'd like me to check these words for any that might be compounded exotics words or deeper the lookup.<|tools_prefix|>[{\"display_answers\": {\"answers\": [\"Mawny, Mawny, Prowly, Snooty, Laughton, Laughtrey, Roughty, Chawner, Jauney, Naughty, Clawny, Shawnee, Hawn, Brawn, Drawn, Farren\"]}}]<|tools_suffix|><|tool_output_start|>\"Answers displayed\"<|tool_output_end|><|assistant_end|>   
"""
    tokenizer = AutoTokenizer.from_pretrained("/capstor/store/cscs/swissai/infra01/reasoning/checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816")
    response_ids = tokenizer.encode(sample_response)
    tool_parser = Apertus2509ToolParser(tokenizer)
    _, function_calls = await tool_parser.extract_tool_calls(response_ids)
    print(function_calls)
    assert len(function_calls) == 2

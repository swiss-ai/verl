# Copyright 2026 The VERL Team and individual contributors.
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


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Placeholder for instruction-following verification.

    The IF verifier for allenai/IF_multi_constraints_upto5 will be implemented
    separately. Returning 0.0 keeps the default reward dispatcher explicit and
    non-crashing for preprocessed IF rows.
    """
    return 0.0

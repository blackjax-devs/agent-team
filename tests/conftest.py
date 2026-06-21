# Copyright 2026- blackjax-devs.
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
# Tests import the installed ``agent_team`` package (CI runs them via
# ``uvx --with-editable . --with pytest pytest``; locally use
# ``uv run --with pytest pytest``). No ``sys.path`` shimming is needed —
# the package is importable from its installed/editable location.

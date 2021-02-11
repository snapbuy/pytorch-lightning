# Copyright The PyTorch Lightning team.
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


class SkipIf(dict):
    """Alow chaining skip if confitions

    >>> from pprint import pprint
    >>> si1 = SkipIf(condition=True, reason="skip-if-1")
    >>> si2 = SkipIf(condition=False, reason="skip-if-2")
    >>> si3 = SkipIf(condition=True, reason="skip-if-3")
    >>> pprint(si1 & si2)
    {'condition': False, 'reason': 'skip-if-1 AND skip-if-2'}
    >>> pprint(si1 + si3)
    {'condition': True, 'reason': 'skip-if-1 AND skip-if-3'}
    >>> pprint(si2 | si3)
    {'condition': True, 'reason': 'skip-if-2 OR skip-if-3'}
    >>> pprint(~si3)
    {'condition': False, 'reason': 'NOT skip-if-3'}
    >>> pprint(si1 and si2)
    {'condition': False, 'reason': 'skip-if-2'}
    >>> pprint(si2 and si3)
    {'condition': True, 'reason': 'skip-if-3'}
    """

    def __and__(self, other: dict) -> dict:
        return SkipIf(condition=self['condition'] and other['condition'], reason=' AND '.join([self['reason'], other['reason']]))

    def __or__(self, other: dict) -> dict:
        return SkipIf(condition=self['condition'] or other['condition'], reason=' OR '.join([self['reason'], other['reason']]))

    def __rand__(self, other: dict) -> dict:
        return self & other

    def __add__(self, other: dict) -> dict:
        return self & other

    def __inv__(self) -> dict:
        return SkipIf(condition=not self['condition'], reason='NOT ' + self['reason'])

    def __invert__(self):
        return self.__inv__()

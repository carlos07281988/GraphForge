# Copyright 2024 GraphForge Contributors
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

"""Pydantic version compatibility layer.

Provides unified access to Pydantic v1 and v2 APIs so that GraphForge
can run with either version installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

_M = TypeVar("_M", bound=BaseModel)

# Detect Pydantic version
try:
    from pydantic import ConfigDict as PydanticConfigDict

    IS_PYDANTIC_V2 = True
except ImportError:
    PydanticConfigDict = dict  # type: ignore[assignment,misc]
    IS_PYDANTIC_V2 = False


def model_dump(model: BaseModel, **kwargs: Any) -> Dict[str, Any]:
    """Serialize a model to a dict, compatible with Pydantic v1 and v2.

    In Pydantic v2, ``model_dump(**kwargs)`` is the canonical method.
    In Pydantic v1, ``dict(**kwargs)`` is the equivalent.
    """
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)  # type: ignore[no-any-return]
    return model.dict(**kwargs)  # type: ignore[no-any-return, union-attr]


def model_copy(
    model: BaseModel,
    *,
    update: Optional[Dict[str, Any]] = None,
    deep: bool = False,
) -> BaseModel:
    """Copy a model with optional updates, compatible with Pydantic v1 and v2.

    In Pydantic v2, ``model_copy(update=..., deep=True)`` is the canonical method.
    In Pydantic v1, ``copy(update=..., deep=True)`` is the equivalent.
    """
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update, deep=deep)  # type: ignore[no-any-return]
    return model.copy(update=update, deep=deep)  # type: ignore[no-any-return, union-attr]


def model_validate(
    model_cls: Type[_M],
    data: Any,
) -> _M:
    """Validate data against a model class, compatible with Pydantic v1 and v2.

    In Pydantic v2, ``model_cls.model_validate(data)`` is the canonical method.
    In Pydantic v1, ``model_cls.parse_obj(data)`` is the equivalent.
    """
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)  # type: ignore[no-any-return, union-attr]


__all__ = [
    "IS_PYDANTIC_V2",
    "PydanticConfigDict",
    "model_dump",
    "model_copy",
    "model_validate",
]

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Narrowing values whose type depends on which constraint a constrained type variable is
//! solved to.

use crate::testcase;

testcase!(
    test_isinstance_narrows_each_case,
    r#"
from typing import TypeVar
class A:
    def m(self) -> int: ...
class C:
    def m(self) -> str: ...
F = TypeVar("F", A, C)
def f(x: F) -> None:
    r = x.m()
    if isinstance(r, int):
        r.bit_length()
        return
    r.upper()
    r.bit_length()  # E: Object of class `str` has no attribute `bit_length`
    "#,
);

testcase!(
    test_is_none_or_isinstance_narrows_each_case,
    r#"
from typing import TypeVar
class A:
    def m(self) -> int | None: ...
class C:
    def m(self) -> str: ...
F = TypeVar("F", A, C)
def f(x: F) -> None:
    q = x.m()
    if q is None or isinstance(q, str):
        return
    q.bit_length()
    q.upper()  # E: Object of class `int` has no attribute `upper`
    "#,
);

testcase!(
    test_truthiness_narrows_each_case,
    r#"
from typing import TypeVar
class A:
    def m(self) -> int | None: ...
class C:
    def m(self) -> str | None: ...
F = TypeVar("F", A, C)
def f(x: F) -> None:
    r = x.m()
    if r:
        r.upper()  # E: Object of class `int` has no attribute `upper`
    if not r:
        return
    r.upper()  # E: Object of class `int` has no attribute `upper`
    "#,
);

testcase!(
    test_is_not_none_keeps_relationship_to_type_variable,
    r#"
from typing import TypeVar
class A:
    def m(self) -> int | None: ...
class C:
    def m(self) -> str | None: ...
F = TypeVar("F", A, C)
def f(x: F) -> None:
    r = x.m()
    if r is not None:
        r.upper()  # E: Object of class `int` has no attribute `upper`
    "#,
);

testcase!(
    test_match_narrows_each_case,
    r#"
from typing import TypeVar
class A:
    def m(self) -> int: ...
class C:
    def m(self) -> str: ...
F = TypeVar("F", A, C)
def f(x: F) -> None:
    r = x.m()
    match r:
        case int():
            r.bit_length()
        case _:
            r.upper()
    "#,
);

testcase!(
    test_independent_values_do_not_narrow_each_other,
    r#"
from typing import TypeVar
class A:
    def m(self) -> int: ...
class C:
    def m(self) -> str: ...
F = TypeVar("F", A, C)
def f(x: F, y: F) -> None:
    r = x.m()
    s = x.m()
    if isinstance(r, int):
        s.bit_length()  # E: Object of class `str` has no attribute `bit_length`
    "#,
);

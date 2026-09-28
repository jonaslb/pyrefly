/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Container results of methods called on values of a constrained type variable, modeled on
//! xarray, pandas, and Polars patterns.

use crate::testcase;

testcase!(
    test_constrained_tuple_result_unpacking,
    r#"
from typing import Self, assert_type
class DataArray:
    def align(self, other: Self) -> tuple[Self, Self]: ...
class Dataset:
    def align(self, other: Self) -> tuple[Self, Self]: ...
def f[T: (DataArray, Dataset)](a: T, b: T) -> tuple[T, T]:
    x, y = a.align(b)
    assert_type(x, T)
    assert_type(y, T)
    return a.align(b)
    "#,
);

testcase!(
    test_constrained_heterogeneous_tuple_result_unpacking,
    r#"
from typing import Self, assert_type
class DataFrame:
    def split(self) -> tuple[int, Self]: ...
class Series:
    def split(self) -> tuple[int, Self]: ...
def f[T: (DataFrame, Series)](x: T) -> T:
    n, rest = x.split()
    assert_type(n, int)
    assert_type(rest, T)
    return rest
    "#,
);

testcase!(
    test_constrained_group_iteration,
    r#"
from typing import Self, assert_type
class DataFrame:
    def groups(self) -> list[tuple[str, Self]]: ...
class Series:
    def groups(self) -> list[tuple[str, Self]]: ...
def f[T: (DataFrame, Series)](x: T) -> list[T]:
    out: list[T] = []
    for group in x.groups():
        assert_type(group, tuple[str, T])
    for key, g in x.groups():
        assert_type(key, str)
        assert_type(g, T)
        out.append(g)
    return out
    "#,
);

testcase!(
    test_constrained_rolling_intermediate,
    r#"
from typing import assert_type
class DataFrame:
    def rolling(self, n: int) -> DataFrameRolling: ...
class Series:
    def rolling(self, n: int) -> SeriesRolling: ...
class DataFrameRolling:
    def mean(self) -> DataFrame: ...
class SeriesRolling:
    def mean(self) -> Series: ...
def f[T: (DataFrame, Series)](x: T) -> T:
    r = x.rolling(3)
    return r.mean()
    "#,
);

testcase!(
    test_constrained_container_negatives,
    r#"
from typing import Self
class DataFrame:
    def split(self) -> tuple[int, Self]: ...
    def groups(self) -> list[tuple[str, Self]]: ...
class Series:
    def split(self) -> tuple[int, Self]: ...
    def groups(self) -> list[tuple[str, Self]]: ...
def wrong_position[T: (DataFrame, Series)](x: T) -> T:
    n, rest = x.split()
    return n  # E: not assignable
def independent[T: (DataFrame, Series), U: (DataFrame, Series)](x: T) -> U:
    _, rest = x.split()
    return rest  # E: not assignable
def wrong_arity[T: (DataFrame, Series)](x: T):
    a, b, c = x.split()  # E: Cannot unpack
def concrete[T: (DataFrame, Series)](x: T) -> T:
    for _, g in x.groups():
        return DataFrame()  # E: not assignable
    return x
    "#,
);

testcase!(
    test_constrained_container_wrong_kind,
    r#"
from typing import Self
class DataFrame:
    def items(self) -> list[Self]: ...
class Series:
    def items(self) -> int: ...
def f[T: (DataFrame, Series)](x: T):
    for _ in x.items():  # E: not iterable
        pass
    "#,
);

testcase!(
    test_constrained_nested_independent_cases,
    r#"
from typing import assert_type
class A: ...
class B: ...
class C: ...
class D: ...
class P[L: (A, B), R: (C, D)]:
    def pair(self) -> tuple[L, R]: ...
def f[L: (A, B), R: (C, D)](p: P[L, R]) -> None:
    l, r = p.pair()
    assert_type(l, L)
    assert_type(r, R)
    "#,
);

// Generic overloads such as `iter` and `next` are evaluated once per constraint when an argument
// depends on the constraint, so the result keeps its cases.
testcase!(
    test_constrained_next_iter,
    r#"
from typing import Self, assert_type
class DataFrame:
    def groups(self) -> list[tuple[str, Self]]: ...
class Series:
    def groups(self) -> list[tuple[str, Self]]: ...
def f[T: (DataFrame, Series)](x: T) -> T:
    assert_type(next(iter(x.groups())), tuple[str, T])
    _, first = next(iter(x.groups()))
    return first
def wrong[T: (DataFrame, Series)](x: T) -> DataFrame:
    _, first = next(iter(x.groups()))
    return first  # E: not assignable to declared return type `DataFrame`
    "#,
);

testcase!(
    test_constrained_different_length_cases,
    r#"
from typing import assert_type
class A:
    def parts(self) -> tuple[int, str]: ...
class B:
    def parts(self) -> tuple[int, str, bytes]: ...
def f[T: (A, B)](x: T):
    a, b = x.parts()  # E: Cannot unpack
    a, b, c = x.parts()  # E: Cannot unpack
    a, b, c, d = x.parts()  # E: Cannot unpack # E: Cannot unpack
    first, *rest = x.parts()
    assert_type(first, int)
    "#,
);

testcase!(
    test_constrained_mixed_fixed_and_unbounded_cases,
    r#"
class A:
    def parts(self) -> tuple[int, int]: ...
class B:
    def parts(self) -> tuple[int, ...]: ...
def f[T: (A, B)](x: T):
    a, b = x.parts()
    a, b, c = x.parts()  # E: Cannot unpack
    "#,
);

testcase!(
    test_constrained_tuple_indexing,
    r#"
from typing import Self, assert_type
class DataArray:
    def align(self, other: Self) -> tuple[Self, Self]: ...
    def split(self) -> tuple[int, Self]: ...
class Dataset:
    def align(self, other: Self) -> tuple[Self, Self]: ...
    def split(self) -> tuple[int, Self]: ...
def f[T: (DataArray, Dataset)](a: T, b: T) -> T:
    pair = a.align(b)
    assert_type(pair[0], T)
    assert_type(pair[-1], T)
    assert_type(a.split()[0], int)
    return pair[1]
    "#,
);

testcase!(
    test_constrained_tuple_indexing_independent_cases,
    r#"
from typing import assert_type
class A: ...
class B: ...
class C: ...
class D: ...
class P[L: (A, B), R: (C, D)]:
    def pair(self) -> tuple[L, R]: ...
def f[L: (A, B), R: (C, D)](p: P[L, R]) -> R:
    assert_type(p.pair()[0], L)
    assert_type(p.pair()[1], R)
    return p.pair()[0]  # E: not assignable
    "#,
);

testcase!(
    test_constrained_tuple_indexing_negatives,
    r#"
from typing import Self
class DataArray:
    def split(self) -> tuple[int, Self]: ...
class Dataset:
    def split(self) -> tuple[int, Self]: ...
def f[T: (DataArray, Dataset)](a: T, i: int) -> T:
    a.split()[2]  # E: out of range
    a.split()[i]
    return a.split()[0]  # E: not assignable
    "#,
);

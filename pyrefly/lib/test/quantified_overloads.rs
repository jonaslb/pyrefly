/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Overload argument type expansion of arguments whose types are constrained type variables.

use crate::testcase;

testcase!(
    test_overloaded_method_independent_quantifieds,
    r#"
from typing import overload
class Series:
    @overload
    def align(self, other: Series) -> tuple[Series, Series]: ...
    @overload
    def align(self, other: DataFrame) -> tuple[Series, DataFrame]: ...
    def align(self, other: Series | DataFrame) -> tuple[Series, Series | DataFrame]: ...
class DataFrame:
    def align[O: (Series, DataFrame)](self, other: O) -> tuple[DataFrame, O]: ...
def mixed[L: (Series, DataFrame), R: (Series, DataFrame)](left: L, right: R) -> tuple[L, R]:
    return left.align(right)
def annotated[L: (Series, DataFrame), R: (Series, DataFrame)](left: L, right: R) -> None:
    x: tuple[L, R] = left.align(right)
def same[F: (Series, DataFrame)](left: F, right: F) -> tuple[F, F]:
    return left.align(right)
def swapped[L: (Series, DataFrame), R: (Series, DataFrame)](left: L, right: R) -> tuple[R, L]:
    return left.align(right)  # E: not assignable to declared return type `tuple[R, L]`
def wrong[L: (Series, DataFrame), R: (Series, DataFrame)](left: L, right: R) -> tuple[L, L]:
    return left.align(right)  # E: not assignable to declared return type `tuple[L, L]`
    "#,
);

testcase!(
    test_overloaded_method_concrete_arguments,
    r#"
from typing import assert_type, overload
class Series:
    @overload
    def align(self, other: Series) -> tuple[Series, Series]: ...
    @overload
    def align(self, other: DataFrame) -> tuple[Series, DataFrame]: ...
    def align(self, other: Series | DataFrame) -> tuple[Series, Series | DataFrame]: ...
class DataFrame: ...
def f(s: Series, df: DataFrame, either: Series | DataFrame) -> None:
    assert_type(s.align(s), tuple[Series, Series])
    assert_type(s.align(df), tuple[Series, DataFrame])
    assert_type(s.align(either), tuple[Series, Series] | tuple[Series, DataFrame])
    "#,
);

testcase!(
    test_overloaded_function_same_quantified_arguments,
    r#"
from typing import overload
class A: ...
class B: ...
@overload
def pair(x: A, y: A) -> A: ...
@overload
def pair(x: B, y: B) -> B: ...
def pair(x: A | B, y: A | B) -> A | B: ...
def same[T: (A, B)](x: T, y: T) -> T:
    return pair(x, y)
def wrong[T: (A, B)](x: T, y: T) -> A:
    return pair(x, y)  # E: not assignable to declared return type `A`
def independent[T: (A, B), U: (A, B)](x: T, y: U) -> T:
    return pair(x, y)  # E: No matching overload found for function `pair`
    "#,
);

testcase!(
    test_overloaded_method_free_quantified_in_signature,
    r#"
from typing import overload
class A: ...
class B: ...
class Box[T]:
    @overload
    def put(self, value: T, other: A) -> tuple[T, A]: ...
    @overload
    def put(self, value: T, other: B) -> tuple[T, B]: ...
    def put(self, value: T, other: A | B) -> tuple[T, A | B]: ...
    @overload
    def local[T2: (A, B)](self, value: T, other: T2, x: A) -> tuple[T, T2]: ...
    @overload
    def local[T2: (A, B)](self, value: T, other: T2, x: B) -> tuple[T, T2]: ...
    def local(self, value: T, other: object, x: A | B) -> tuple[T, object]: ...
def f[F: (A, B), R: (A, B)](box: Box[F], value: F, other: R) -> tuple[F, R]:
    return box.put(value, other)
def g[F: (A, B), R: (A, B)](box: Box[F], value: F, other: R) -> tuple[R, F]:
    return box.put(value, other)  # E: not assignable to declared return type `tuple[R, F]`
def h[F: (A, B), R: (A, B)](box: Box[F], value: F, other: R, x: F) -> tuple[F, R]:
    return box.local(value, other, x)
    "#,
);

// After splitting `T`, the second argument is `int | str` for `A` but `bytes` for `B`, so each
// list must expand its own argument type.
testcase!(
    test_overload_expansion_per_constraint_argument_types,
    r#"
from typing import overload
class A:
    def value(self) -> int | str: ...
class B:
    def value(self) -> bytes: ...
@overload
def bad(x: A, value: int) -> int: ...
@overload
def bad(x: A, value: str) -> int: ...
@overload
def bad(x: B, value: int | str) -> int: ...
def bad(x: A | B, value: int | str | bytes) -> int: ...
@overload
def good(x: A, value: int) -> int: ...
@overload
def good(x: A, value: str) -> str: ...
@overload
def good(x: B, value: bytes) -> bytes: ...
def good(x: A | B, value: int | str | bytes) -> int | str | bytes: ...
def invalid[T: (A, B)](x: T) -> int:
    return bad(x, x.value())  # E: No matching overload found for function `bad`
def valid[T: (A, B)](x: T) -> int | str | bytes:
    return good(x, x.value())
def valid_precise[T: (A, B)](x: T) -> bytes:
    return good(x, x.value())  # E: not assignable to declared return type `bytes`
    "#,
);

// After splitting `T`, `y` is `int | None` under `T = int` but only `None` under `T = None`, so
// the branches expand to different widths.
testcase!(
    test_overload_expansion_irregular_widths,
    r#"
from typing import overload
@overload
def f(x: int, y: int) -> int: ...
@overload
def f(x: int, y: None) -> bytes: ...
@overload
def f(x: None, y: None) -> str: ...
def f(x: int | None, y: int | None) -> int | bytes | str: ...
def ok[T: (int, None)](x: T, y: T | None) -> int | bytes | str:
    return f(x, y)
def wrong[T: (int, None)](x: T, y: T | None) -> int | bytes:
    return f(x, y)  # E: not assignable to declared return type `bytes | int`
    "#,
);

testcase!(
    test_overload_expansion_independent_quantifieds_and_union,
    r#"
from typing import overload
class A: ...
class B: ...
@overload
def g(x: A, y: A) -> tuple[A, A]: ...
@overload
def g(x: A, y: B) -> tuple[A, B]: ...
@overload
def g(x: B, y: A) -> tuple[B, A]: ...
@overload
def g(x: B, y: B) -> tuple[B, B]: ...
def g(x: A | B, y: A | B) -> tuple[A | B, A | B]: ...
def ok[T: (A, B), U: (A, B)](x: T, y: U) -> tuple[T, U]:
    return g(x, y)
def swapped[T: (A, B), U: (A, B)](x: T, y: U) -> tuple[U, T]:
    return g(x, y)  # E: not assignable to declared return type `tuple[U, T]`
def union_first[U: (A, B)](x: A | B, y: U) -> tuple[A | B, U]:
    return g(x, y)
def union_not_correlated[U: (A, B)](x: A | B, y: U) -> tuple[U, U]:
    return g(x, y)  # E: not assignable to declared return type `tuple[U, U]`
    "#,
);

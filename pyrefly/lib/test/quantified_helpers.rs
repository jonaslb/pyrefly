/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Inferred returns of generic helpers over constrained type variables keep their
//! per-constraint results across call boundaries.

use crate::testcase;

testcase!(
    test_case_results_preserve_invariant_containers,
    r#"
class A:
    def items(self) -> list[A]: ...
class B:
    def items(self) -> list[B]: ...
def preserve[T: (A, B)](x: T) -> list[T]:
    return x.items()
def cannot_widen[T: (A, B)](x: T) -> list[A | B]:
    return x.items()  # E: not assignable
def cannot_invent[T: (A, B)](x: list[A] | list[B]) -> list[T]:
    return x  # E: not assignable
"#,
);

testcase!(
    test_helper_return_specialized_at_call_site,
    r#"
from typing import assert_type
class DataFrame:
    def group_by(self) -> DataFrameGroup: ...
class LazyFrame:
    def group_by(self) -> LazyFrameGroup: ...
class DataFrameGroup:
    def agg(self) -> DataFrame: ...
class LazyFrameGroup:
    def agg(self) -> LazyFrame: ...
def grouped[F: (DataFrame, LazyFrame)](x: F):
    return x.group_by()
assert_type(grouped(DataFrame()), DataFrameGroup)
assert_type(grouped(LazyFrame()), LazyFrameGroup)
assert_type(grouped(DataFrame()).agg(), DataFrame)
assert_type(grouped(LazyFrame()).agg(), LazyFrame)
    "#,
);

testcase!(
    test_helper_calls_do_not_contaminate_each_other,
    r#"
from typing import assert_type
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
def helper[F: (A, B)](x: F):
    return x.m()
a = helper(A())
b = helper(B())
assert_type(a, int)
assert_type(b, str)
assert_type((helper(A()), helper(B())), tuple[int, str])
    "#,
);

testcase!(
    test_forwarding_preserves_quantified,
    r#"
from typing import assert_type
class DataFrame:
    def group_by(self) -> DataFrameGroup: ...
class LazyFrame:
    def group_by(self) -> LazyFrameGroup: ...
class DataFrameGroup:
    def agg(self) -> DataFrame: ...
class LazyFrameGroup:
    def agg(self) -> LazyFrame: ...
def grouped[F: (DataFrame, LazyFrame)](x: F):
    return x.group_by()
def summarize[G: (DataFrame, LazyFrame)](x: G) -> G:
    return grouped(x).agg()
def summarize_inferred[G: (DataFrame, LazyFrame)](x: G):
    return grouped(x).agg()
assert_type(summarize_inferred(DataFrame()), DataFrame)
assert_type(summarize_inferred(LazyFrame()), LazyFrame)
    "#,
);

testcase!(
    test_forwarding_with_reordered_constraints,
    r#"
from typing import assert_type
class DataFrame:
    def group_by(self) -> DataFrameGroup: ...
class LazyFrame:
    def group_by(self) -> LazyFrameGroup: ...
class Dataset:
    def group_by(self) -> DatasetGroup: ...
class DataFrameGroup:
    def agg(self) -> DataFrame: ...
class LazyFrameGroup:
    def agg(self) -> LazyFrame: ...
class DatasetGroup:
    def agg(self) -> Dataset: ...
def grouped[F: (DataFrame, LazyFrame, Dataset)](x: F):
    return x.group_by()
def reordered[G: (Dataset, LazyFrame, DataFrame)](x: G) -> G:
    return grouped(x).agg()
def subset[G: (Dataset, DataFrame)](x: G) -> G:
    return grouped(x).agg()
def subset_inferred[G: (Dataset, DataFrame)](x: G):
    return grouped(x)
assert_type(subset_inferred(Dataset()), DatasetGroup)
assert_type(subset_inferred(DataFrame()), DataFrameGroup)
def subset_wrong[G: (Dataset, DataFrame)](x: G) -> G:
    return grouped(x)  # E: not assignable
    "#,
);

testcase!(
    test_forwarding_with_unrelated_constraints_is_rejected,
    r#"
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
class C(A): ...
def helper[F: (A, B)](x: F):
    return x.m()
def forward[G: (A, B)](x: G) -> int:
    return helper(x)  # E: not assignable
def wrong[G: (A, B)](x: G) -> G:
    return helper(x)  # E: not assignable
    "#,
);

testcase!(
    test_helper_with_any_argument_is_conservative,
    r#"
from typing import Any, assert_type
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
def helper[F: (A, B)](x: F):
    return x.m()
def f(x: Any):
    assert_type(helper(x), int | str)
    "#,
);

testcase!(
    test_helper_return_as_external_callable,
    r#"
from typing import Callable, assert_type, overload
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
def helper[F: (A, B)](x: F):
    return x.m()
def apply[T, R](f: Callable[[T], R], x: T) -> R: ...
ok: Callable[[A], int] = helper
ok_b: Callable[[B], str] = helper
bad: Callable[[A], str] = helper  # E: not assignable
bad_b: Callable[[B], int] = helper  # E: not assignable
assert_type(apply(helper, A()), int)
assert_type(apply(helper, B()), str)
@overload
def overloaded_apply(f: Callable[[A], int], x: A) -> int: ...
@overload
def overloaded_apply(f: Callable[[B], str], x: B) -> str: ...
def overloaded_apply(f, x) -> int | str: ...
assert_type(overloaded_apply(helper, A()), int)
assert_type(overloaded_apply(helper, B()), str)
def accept_union(f: Callable[[A], bytes] | Callable[[B], str]) -> None: ...
# A failed callable alternative must not retain its pending-selector bounds.
accept_union(helper)
assert_type(apply(helper, B()), str)
# Each assignment instantiates the helper afresh, so one choice does not leak into the next.
def pair() -> tuple[Callable[[A], int], Callable[[B], str]]:
    return (helper, helper)
    "#,
);

testcase!(
    test_helper_wrong_constraint_argument,
    r#"
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
def helper[F: (A, B)](x: F):
    return x.m()
helper(1)  # E:
x: str = helper(A())  # E: not assignable
    "#,
);

testcase!(
    test_repeated_call_sites_choose_independently,
    r#"
from typing import assert_type
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
def helper[F: (A, B)](x: F):
    return x.m()
def use(a: A, b: B):
    for _ in range(3):
        assert_type(helper(a), int)
        assert_type(helper(b), str)
        assert_type(helper(a), int)
    xs = [helper(a), helper(b)]
    assert_type(xs, list[int | str])
    "#,
);

testcase!(
    test_subclass_argument_selects_constraint_case,
    r#"
from typing import assert_type
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
class C(A): ...
def helper[F: (A, B)](x: F):
    return x.m()
assert_type(helper(C()), int)
# Realignment matches constraints by equality, so `C` does not map to the `A` case.
def g[G: (B, C)](x: G):
    return helper(x)
assert_type(g(C()), int | str)
assert_type(g(B()), int | str)
    "#,
);

testcase!(
    test_rolling_via_inferred_intermediate_helper,
    r#"
from typing import assert_type
class DataFrame:
    def rolling(self) -> DataFrameRolling: ...
class Dataset:
    def rolling(self) -> DatasetRolling: ...
class DataFrameRolling:
    def mean(self) -> DataFrame: ...
class DatasetRolling:
    def mean(self) -> Dataset: ...
def windowed[F: (DataFrame, Dataset)](x: F):
    return x.rolling()
w = windowed(Dataset())
assert_type(w, DatasetRolling)
assert_type(w.mean(), Dataset)
assert_type(windowed(DataFrame()).mean(), DataFrame)
def reversed_forward[G: (Dataset, DataFrame)](x: G) -> G:
    r = windowed(x)
    return r.mean()
    "#,
);

testcase!(
    test_overlapping_constraints_select_from_lower_bounds,
    r#"
from typing import Callable, Self, assert_type
class A:
    def m(self) -> list[Self]: ...
class B(A): ...
def helper[F: (A, B)](x: F):
    return x.m()
assert_type(helper(A()), list[A])
assert_type(helper(B()), list[B])
ok_a: Callable[[A], list[A]] = helper
ok_b: Callable[[B], list[B]] = helper
bad: Callable[[A], list[B]] = helper  # E: not assignable
    "#,
);

testcase!(
    test_gradual_parameters_do_not_select_a_case,
    r#"
from typing import Any, Callable, Never
class A:
    def m(self) -> int: ...
class B:
    def m(self) -> str: ...
def helper[F: (A, B)](x: F):
    return x.m()
any_param: Callable[[Any], int | str] = helper
never_param: Callable[[Never], int | str] = helper
    "#,
);

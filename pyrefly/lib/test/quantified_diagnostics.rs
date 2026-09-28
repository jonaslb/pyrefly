/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Diagnostics produced while checking each constraint of a constrained type
//! variable are reported once for the original expression.

use crate::testcase;

testcase!(
    test_missing_attr_on_all_constraints_reported_once,
    r#"
def f[S: (int, str)](x: S):
    x.bogus  # E: Object of type `S` has no attribute `bogus`
    "#,
);

testcase!(
    test_missing_attr_on_one_constraint,
    r#"
class A:
    x: int
class B: ...
def f[S: (A, B)](v: S):
    v.x  # E: Object of class `B` has no attribute `x`
    "#,
);

testcase!(
    test_errors_at_different_positions_not_merged,
    r#"
def f[S: (int, str)](x: S):
    x.bogus  # E: Object of type `S` has no attribute `bogus`
    x.other  # E: Object of type `S` has no attribute `other`
    "#,
);

testcase!(
    test_ordinary_union_diagnostic_unchanged,
    r#"
def f(x: int | str):
    x.bogus  # E: Object of type `int | str` has no attribute `bogus`
    "#,
);

testcase!(
    test_call_errors_merge_by_argument_position,
    r#"
class A:
    def method(self, first: int, second: int) -> None: ...
class B:
    def method(self, first: str, second: str) -> None: ...
def f[T: (A, B)](x: T):
    x.method(None, None)  # E: Call is invalid for some constraints # E: Call is invalid for some constraints
"#,
);

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Callbacks passed to methods of a value whose type is a constrained type variable.

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_lambda_callback_per_constraint,
    r#"
from typing import Callable, assert_type
class A:
    def apply(self, f: Callable[[A], A]) -> A: ...
    def pipe[R](self, f: Callable[[A], R]) -> R: ...
    def join(self, other: A) -> A: ...
    def mutated(self) -> A: ...
    def only_a(self) -> A: ...
class B:
    def apply(self, f: Callable[[B], B]) -> B: ...
    def pipe[R](self, f: Callable[[B], R]) -> R: ...
    def join(self, other: B) -> B: ...
    def mutated(self) -> B: ...
def mutate_a(x: A) -> A: ...
def identity[X](x: X) -> X: ...

def apply[T: (A, B)](x: T) -> T:
    return x.apply(lambda y: y.mutated())
def pipe[T: (A, B)](x: T) -> T:
    return x.pipe(lambda y: y.mutated())
def pipe_constant[T: (A, B)](x: T) -> int:
    return x.pipe(lambda y: 1)
def named_generic[T: (A, B)](x: T) -> T:
    return x.apply(identity)
def named_concrete[T: (A, B)](x: T) -> T:
    return x.apply(mutate_a)  # E: Argument `(x: A) -> A` is not assignable to parameter `f` with type `(B) -> B`
def body_error[T: (A, B)](x: T) -> T:
    return x.apply(lambda y: y.only_a())  # E: Object of class `B` has no attribute `only_a`
def composed[T: (A, B)](x: T, y: T) -> T:
    return x.apply(lambda z: z.mutated()).join(y.apply(lambda z: z.mutated()))
 "#,
);

testcase!(
    test_lambda_captures_same_quantified,
    r#"
from typing import Callable, assert_type
class A:
    def apply(self, f: Callable[[A], A]) -> A: ...
    def join(self, other: A) -> A: ...
class B:
    def apply(self, f: Callable[[B], B]) -> B: ...
    def join(self, other: B) -> B: ...

def same[T: (A, B)](left: T, right: T) -> T:
    out = left.apply(lambda x: x.join(right))
    assert_type(right, T)
    return out
def independent[T: (A, B), U: (A, B)](left: T, right: U) -> T:
    return left.apply(lambda x: x.join(right))  # E: Call is invalid for some constraints of `T`
 "#,
);

// Keys inside a lambda body (comprehensions, assignment expressions, narrowing, nested lambdas)
// are cached, so they cannot be re-inferred for each constraint. Such a lambda is inferred once
// with the union of its parameter types across constraints, in either constraint order, rather
// than checked against a stale answer.
testcase!(
    test_lambda_with_cached_body_keys,
    r#"
from typing import Callable
class A:
    def pipe[R](self, f: Callable[[A], R]) -> R: ...
class B:
    def pipe[R](self, f: Callable[[B], R]) -> R: ...

def comprehension_ab[T: (A, B)](x: T) -> A:
    return x.pipe(lambda arg: [i for i in [arg]][0])  # E: `A | B` is not assignable to declared return type `A`
def comprehension_ba[T: (B, A)](x: T) -> A:
    return x.pipe(lambda arg: [i for i in [arg]][0])  # E: `A | B` is not assignable to declared return type `A`
def generator[T: (A, B)](x: T) -> A:
    return x.pipe(lambda arg: next(i for i in [arg]))  # E: `A | B` is not assignable to declared return type `A`
def walrus_ab[T: (A, B)](x: T) -> A:
    return x.pipe(lambda arg: (saved := arg))  # E: `A | B` is not assignable to declared return type `A`
def walrus_ba[T: (B, A)](x: T) -> A:
    return x.pipe(lambda arg: (saved := arg))  # E: `A | B` is not assignable to declared return type `A`
def walrus_tuple[T: (A, B)](x: T) -> tuple[T, A]:
    return x.pipe(lambda arg: ((saved := arg), saved))  # E: `tuple[A | B, A | B]` is not assignable
def ternary[T: (A, B)](x: T) -> A:
    return x.pipe(lambda arg: arg if arg else arg)  # E: `A | B` is not assignable to declared return type `A`
def boolop_ab[T: (A, B)](x: T) -> A:
    return x.pipe(lambda arg: arg and arg)  # E: `A | B` is not assignable to declared return type `A`
def boolop_ba[T: (B, A)](x: T) -> A:
    return x.pipe(lambda arg: arg or arg)  # E: `A | B` is not assignable to declared return type `A`
def nested_lambda[T: (A, B)](x: T) -> A:
    return x.pipe(lambda arg: (lambda: [i for i in [arg]][0])())  # E: `A | B` is not assignable to declared return type `A`
def in_list[T: (A, B)](x: T) -> list[Callable[[A], A]]:
    return [x.pipe(lambda arg: [lambda: [i for i in [arg]][0]])]  # E: `list[list[() -> A | B]]` is not assignable
 "#,
);

// The generic function's return type is inferred when `main` calls it, so the lambda is first
// solved on behalf of another module. The result does not depend on which case is solved first.
testcase!(
    test_lambda_with_cached_body_keys_cross_module,
    TestEnv::one(
        "lib",
        r#"
from typing import Callable
class A:
    def pipe[R](self, f: Callable[[A], R]) -> R: ...
class B:
    def pipe[R](self, f: Callable[[B], R]) -> R: ...
def f[T: (A, B)](x: T):
    return x.pipe(lambda arg: [i for i in [arg]][0])
def g[T: (B, A)](x: T):
    return x.pipe(lambda arg: (saved := arg))
"#,
    ),
    r#"
from typing import assert_type
from lib import A, B, f, g
assert_type(f(A()), A | B)
assert_type(f(B()), A | B)
assert_type(g(A()), A | B)
assert_type(g(B()), A | B)
 "#,
);

// These are valid, but their lambda bodies create cached keys that depend on the lambda's
// parameters, so they are inferred once with a union parameter type, which loses the
// correlation between the parameter and `T`.
testcase!(
    bug = "Lambdas with cached body keys lose their correlation with `T`",
    test_lambda_with_cached_body_keys_valid,
    r#"
from typing import Callable
class A:
    def pipe[R](self, f: Callable[[A], R]) -> R: ...
class B:
    def pipe[R](self, f: Callable[[B], R]) -> R: ...

def comprehension[T: (A, B)](x: T) -> T:
    return x.pipe(lambda arg: [i for i in [arg]][0])  # E: `A | B` is not assignable to declared return type `T`
def walrus[T: (A, B)](x: T) -> T:
    return x.pipe(lambda arg: (saved := arg))  # E: `A | B` is not assignable to declared return type `T`
 "#,
);

// A sort key whose body narrows its parameter accepts the union of the element types, which
// is valid for each constraint.
testcase!(
    test_sort_key_with_cached_body_keys,
    r#"
def f[T: (list[int], list[str])](x: T) -> None:
    x.sort(key=lambda e: e or 0)
    x.sort(key=lambda e: e if isinstance(e, int) else len(e))
def g[T: (list[str], list[int])](x: T) -> None:
    x.sort(key=lambda e: e if isinstance(e, int) else len(e))
    x.sort(key=lambda e: e or 0)
def h[T: (list[int], list[str])](x: T) -> None:
    x.sort(key=lambda e: e if isinstance(e, int) else e.upper)  # E: Call is invalid for some constraints of `T`
 "#,
);

// Probing each constraint for the contextual type of a lambda with cached body keys is
// speculative, so the probes do not pin the element type of `list()`. Otherwise, the probe for
// `B` would fail after the probe for `A`, and the lambda would be rejected. Only the calls that
// check the inferred lambda pin it, and the second of them then fails.
testcase!(
    test_lambda_with_cached_body_keys_probe_is_rolled_back,
    r#"
from typing import Callable
class A:
    def pipe[R](self, xs: list[int], f: Callable[[A], R]) -> R: ...
class B:
    def pipe[R](self, xs: list[str], f: Callable[[B], R]) -> R: ...
def f[T: (A, B)](x: T) -> None:
    x.pipe(list(), lambda arg: [i for i in [arg]][0])  # E: `list[int]` is not assignable to parameter `xs` with type `list[str]`
 "#,
);

// Probes only evaluate argument types, so a lambda with cached body keys is not inferred once
// for all constraints when another argument is still contextually typed.
testcase!(
    bug = "The lambda could be inferred once if the probes also typed the list literal",
    test_lambda_with_cached_body_keys_beside_contextual_argument,
    r#"
from typing import Callable
class A:
    def pipe[R](self, xs: list[int], f: Callable[[A], R]) -> R: ...
class B:
    def pipe[R](self, xs: list[int], f: Callable[[B], R]) -> R: ...
def f[T: (A, B)](x: T) -> None:
    x.pipe([], lambda arg: [i for i in [arg]][0])  # E: Cannot check this argument separately
 "#,
);

// Keyword-only and defaulted lambda parameters keep their kinds in the contextual type.
testcase!(
    test_lambda_with_cached_body_keys_parameter_kinds,
    r#"
from typing import Callable, Protocol
class KeyOnly(Protocol):
    def __call__(self, *, e: int) -> object: ...
class KeyOnlyStr(Protocol):
    def __call__(self, *, e: str) -> object: ...
class A:
    def pipe(self, f: KeyOnly) -> None: ...
class B:
    def pipe(self, f: KeyOnlyStr) -> None: ...
def f[T: (A, B)](x: T) -> None:
    x.pipe(lambda *, e: e or 0)
    x.pipe(lambda *, e, extra=0: e or extra)
 "#,
);

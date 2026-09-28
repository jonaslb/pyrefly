/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A type whose value depends on which declared constraint a constrained type variable is
//! solved to.
//!
//! Attribute lookup and calls introduce cases from a constrained quantified. Other operations
//! project an existing case value; they need not expand every bare TypeVar. Projection uses
//! `with_cases` to retain either the live quantified or its pending call-site selector.

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;

use crate::quantified::Quantified;
use crate::type_var::Restriction;
use crate::types::Type;

/// The maximum number of leaf alternatives in a nested tree of `QuantifiedCases`. Larger trees
/// are conservatively erased to the union of their leaves.
const MAX_LEAF_CASES: usize = 64;

/// The result of a computation performed once per declared constraint of `quantified`.
///
/// Invariant: `quantified` has `Restriction::Constraints(cs)` and `cases.len() == cs.len()`,
/// where `cases[i]` is the result when `quantified` is solved to `cs[i]`. Entries are aligned with
/// the declaration order of the constraints, so equal entries are preserved. A case never
/// contains a free `QuantifiedCases` over the same `quantified`, because that has already been
/// specialized to the matching index.
///
/// When `selector` is `Some`, `quantified` has been substituted at a call boundary by a solver
/// variable, and the cases are selected by whatever that variable is solved to. `quantified` then
/// only supplies the declared constraint list and is not a free occurrence of the type variable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct QuantifiedCases {
    quantified: Quantified,
    cases: Vec<Type>,
    selector: Option<Box<Type>>,
}

fn constraints(q: &Quantified) -> &[Type] {
    match q.restriction() {
        Restriction::Constraints(cs) => cs,
        _ => panic!("QuantifiedCases requires a constrained type variable"),
    }
}

impl QuantifiedCases {
    /// The constrained type variable that selects the case, or `None` while the choice is pending
    /// on a solver variable. A pending value's declaration quantified is not a live free
    /// occurrence, so it must not be used to specialize or rebuild anything.
    pub fn quantified(&self) -> Option<&Quantified> {
        match self.selector {
            None => Some(&self.quantified),
            Some(_) => None,
        }
    }

    pub fn cases(&self) -> &[Type] {
        &self.cases
    }

    /// The declared constraints that the cases are aligned with. This is available even while
    /// the choice is pending, because it is only a list of types and not a live quantified.
    pub fn constraints(&self) -> &[Type] {
        constraints(&self.quantified)
    }

    /// The type standing in for the declaration quantified that selects the case, if that
    /// quantified has been substituted by a solver variable.
    pub fn selector(&self) -> Option<&Type> {
        self.selector.as_deref()
    }

    /// Rebuilds these cases with new per-constraint results, keeping the selector.
    pub fn with_cases(&self, cases: Vec<Type>) -> Type {
        Type::build_quantified_cases(self.quantified.clone(), cases, self.selector.clone())
    }

    /// Chooses the result once the selector is solved to `replacement`.
    ///
    /// - A solver variable defers the choice until the variable is solved.
    /// - A constrained quantified with the same constraint list keeps the alignment. One whose
    ///   constraints each equal exactly one of ours (in any order, or a subset) realigns the cases.
    /// - A replacement equal to declared constraints selects the matching cases.
    /// - Anything else, including `Any` and ambiguous realignments, erases to the union of cases.
    pub fn select(&self, replacement: Type) -> Type {
        let cs = constraints(&self.quantified);
        let matching = |c: &Type| -> Vec<usize> {
            cs.iter()
                .enumerate()
                .filter_map(|(i, x)| (x == c).then_some(i))
                .collect()
        };
        match replacement {
            Type::Var(_) => Type::build_quantified_cases(
                self.quantified.clone(),
                self.cases.clone(),
                Some(Box::new(replacement)),
            ),
            Type::Quantified(q2) if matches!(q2.restriction(), Restriction::Constraints(_)) => {
                let cs2 = constraints(&q2);
                let realigned: Option<Vec<Type>> = if cs2 == cs {
                    Some(self.cases.clone())
                } else {
                    cs2.iter()
                        .map(|c| match matching(c).as_slice() {
                            [i] => Some(self.cases[*i].clone()),
                            _ => None,
                        })
                        .collect()
                };
                match realigned {
                    Some(cases) if !cases.is_empty() => Type::quantified_cases(*q2, cases),
                    _ => self.erase(),
                }
            }
            replacement => {
                let mut matches: Vec<Type> = matching(&replacement)
                    .into_iter()
                    .map(|i| self.cases[i].clone())
                    .collect();
                match matches.len() {
                    0 => self.erase(),
                    1 => matches.pop().expect("there is one match"),
                    _ => Type::union(matches),
                }
            }
        }
    }

    /// Resolves the cases once the selector is solved to something other than a variable.
    pub fn resolve_selector(&self) -> Option<Type> {
        match self.selector.as_deref() {
            Some(Type::Var(_)) | None => None,
            Some(t) => Some(self.select(t.clone())),
        }
    }

    pub(crate) fn recurse_cases_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        self.cases.visit_mut(f);
    }

    /// Splits cases selected by a live quantified into that quantified and its per-constraint
    /// results. A pending value cannot be split: it yields `Err` with its resolved result if the
    /// selector has been solved, and otherwise with the union of its cases, which is safe for
    /// consumers that cannot carry the selector.
    pub fn into_selected_parts(self) -> Result<(Quantified, Vec<Type>), Type> {
        match self.selector {
            None => Ok((self.quantified, self.cases)),
            Some(_) => Err(self.resolve_selector().unwrap_or_else(|| self.erase())),
        }
    }

    /// The union of every leaf result, forgetting which constraint selects which result.
    pub fn erase(&self) -> Type {
        let mut leaves = Vec::new();
        for case in &self.cases {
            case.clone()
                .erase_quantified_cases()
                .push_union_members(&mut leaves);
        }
        match leaves.len() {
            1 => leaves.pop().expect("the union has one member"),
            _ => Type::union(leaves),
        }
    }
}

impl Type {
    /// Views a constrained quantified or a dependent result as constraint-indexed cases. Cases
    /// whose choice is pending on a solver variable have no live quantified and are not viewed.
    pub fn as_quantified_cases(&self) -> Option<(&Quantified, &[Type])> {
        match self {
            Type::Quantified(q) => match q.restriction() {
                Restriction::Constraints(cases) => Some((q, cases)),
                _ => None,
            },
            Type::QuantifiedCases(cases) => Some((cases.quantified()?, cases.cases())),
            _ => None,
        }
    }

    /// Creates the per-constraint result for the constrained type variable `q`, where `cases[i]`
    /// corresponds to the `i`-th declared constraint.
    ///
    /// The result collapses to `q` when every case is its own constraint, and to the shared value
    /// when all cases are equal. Nested cases over `q` are specialized to the matching index, and
    /// trees exceeding the leaf budget are erased to a union.
    pub fn quantified_cases(q: Quantified, cases: Vec<Type>) -> Type {
        Type::build_quantified_cases(q, cases, None)
    }

    fn build_quantified_cases(
        q: Quantified,
        cases: Vec<Type>,
        selector: Option<Box<Type>>,
    ) -> Type {
        if let Some(solved) = selector.as_deref()
            && !matches!(solved, Type::Var(_))
        {
            return QuantifiedCases {
                quantified: q,
                cases,
                selector: None,
            }
            .select(solved.clone());
        }
        let cs = constraints(&q);
        assert!(
            !cs.is_empty(),
            "QuantifiedCases requires at least one constraint"
        );
        assert_eq!(
            cs.len(),
            cases.len(),
            "QuantifiedCases must have one case per declared constraint"
        );
        // `q` is only free in the cases when it selects them.
        let cases: Vec<Type> = match selector {
            None => cases
                .iter()
                .enumerate()
                .map(|(i, case)| case.specialize_quantified(&q, i))
                .collect(),
            Some(_) => cases,
        };
        if cases.iter().zip(cs).all(|(case, c)| case == c) {
            return match selector {
                Some(selector) => *selector,
                None => Type::Quantified(Box::new(q)),
            };
        }
        if cases.iter().all(|case| case == &cases[0]) {
            return cases.into_iter().next().expect("cases are nonempty");
        }
        let result = QuantifiedCases {
            quantified: q,
            cases,
            selector,
        };
        if result
            .cases
            .iter()
            .map(Type::leaf_case_count)
            .sum::<usize>()
            > MAX_LEAF_CASES
        {
            result.erase()
        } else {
            Type::QuantifiedCases(Box::new(result))
        }
    }

    /// Selects the `index`-th case of every `QuantifiedCases` over `q`, and substitutes free
    /// occurrences of `q` with its `index`-th declared constraint. Occurrences bound by an inner
    /// `Forall` are left alone.
    pub fn specialize_quantified(&self, q: &Quantified, index: usize) -> Type {
        fn specialize(
            ty: &mut Type,
            q: &Quantified,
            index: usize,
            bound: &mut Vec<Quantified>,
        ) -> bool {
            if bound.contains(q) {
                return false;
            }
            let mut changed = false;
            if let Type::QuantifiedCases(cases) = ty
                && let Some(resolved) = cases.resolve_selector()
            {
                *ty = resolved;
                changed = true;
            }
            match ty {
                Type::Quantified(candidate) if candidate.as_ref() == q => {
                    *ty = constraints(q)[index].clone();
                    changed = true;
                }
                Type::QuantifiedCases(cases) if cases.quantified() == Some(q) => {
                    *ty = cases.cases()[index].clone();
                    specialize(ty, q, index, bound);
                    changed = true;
                }
                // The selector is absent or an unsolved variable, which `q` cannot occur in.
                Type::QuantifiedCases(cases) => {
                    cases.recurse_cases_mut(&mut |ty| {
                        changed |= specialize(ty, q, index, bound);
                    });
                    if changed {
                        *ty = cases.with_cases(cases.cases.clone());
                    }
                }
                _ => ty.recurse_with_type_parameter_scopes_mut(bound, &mut |ty, bound| {
                    changed |= specialize(ty, q, index, bound);
                }),
            }
            changed
        }
        assert!(index < constraints(q).len(), "constraint index is in range");
        let mut ty = self.clone();
        specialize(&mut ty, q, index, &mut Vec::new());
        ty
    }

    /// Replaces every `QuantifiedCases` with the union of its leaves.
    pub fn erase_quantified_cases(mut self) -> Type {
        self.transform_mut(&mut |ty| {
            if let Type::QuantifiedCases(x) = ty {
                *ty = x.erase();
            }
        });
        self
    }

    fn leaf_case_count(&self) -> usize {
        match self {
            Type::QuantifiedCases(x) => x.cases.iter().map(Type::leaf_case_count).sum(),
            _ => {
                // Choices in separate type arguments can be independent. Counting
                // their product is conservative when they share a quantified.
                let mut n = 1usize;
                self.recurse(&mut |t| {
                    n = n
                        .saturating_mul(t.leaf_case_count())
                        .min(MAX_LEAF_CASES + 1);
                });
                n
            }
        }
    }

    fn push_union_members(self, out: &mut Vec<Type>) {
        match self {
            Type::Union(u) => {
                for m in u.members {
                    if !out.contains(&m) {
                        out.push(m);
                    }
                }
            }
            t => {
                if !out.contains(&t) {
                    out.push(t);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyrefly_python::module_name::ModuleName;
    use pyrefly_util::uniques::UniqueFactory;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use crate::callable::Callable;
    use crate::callable::Params;
    use crate::dimension::Int;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedOrigin;
    use crate::tuple::Tuple;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::Forallable;
    use crate::types::TParams;
    use crate::types::Type;
    use crate::types::Var;
    fn q(name: &str, offset: u32, constraints: Vec<Type>) -> Quantified {
        let identity = QuantifiedIdentity::new(
            ModuleName::from_str("test"),
            AnchorIndex::first(TextRange::empty(TextSize::new(offset))),
            QuantifiedOrigin::synthetic(),
        );
        Quantified::type_var(
            Name::new(name),
            identity,
            None,
            Restriction::Constraints(constraints),
            PreInferenceVariance::Undefined,
        )
    }

    fn lit(n: i64) -> Type {
        Type::Int(Int::Literal(n))
    }

    fn tq(x: &Quantified) -> Type {
        Type::Quantified(Box::new(x.clone()))
    }

    #[test]
    fn test_collapse() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        assert_eq!(
            Type::quantified_cases(t.clone(), vec![Type::None, lit(1)]),
            tq(&t)
        );
        assert_eq!(
            Type::quantified_cases(t.clone(), vec![lit(2), lit(2)]),
            lit(2)
        );
    }

    #[test]
    fn test_preserves_order_and_duplicates() {
        let t = q("T", 0, vec![Type::None, lit(1), lit(2)]);
        let ty = Type::quantified_cases(t.clone(), vec![lit(5), lit(5), lit(6)]);
        let Type::QuantifiedCases(x) = &ty else {
            panic!("expected cases")
        };
        assert_eq!(x.cases(), &[lit(5), lit(5), lit(6)]);
        assert_eq!(ty.specialize_quantified(&t, 1), lit(5));
        assert_eq!(ty.specialize_quantified(&t, 2), lit(6));
    }

    #[test]
    fn test_duplicate_constraints_use_explicit_index() {
        let t = q("T", 0, vec![lit(1), lit(1)]);
        let ty = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)]);
        assert_eq!(ty.specialize_quantified(&t, 1), lit(6));
        let mut substituted = ty;
        substituted.subst_mut_fn(&mut |x| (x == &t).then(|| lit(1)));
        assert_eq!(substituted, Type::union(vec![lit(5), lit(6)]));
    }

    #[test]
    fn test_specialization_respects_binders() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let cases = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)]);
        let function = Forallable::Callable(Callable {
            params: Params::Ellipsis,
            ret: cases,
        })
        .forall(Arc::new(TParams::new(vec![t.clone()])));
        assert_eq!(function.specialize_quantified(&t, 0), function);
        let mut substituted = function.clone();
        substituted.subst_mut_fn(&mut |x| (x == &t).then_some(Type::None));
        assert_eq!(substituted, function);
        let mut free = Vec::new();
        function.for_each_free_quantified(&mut |q| free.push(q));
        assert!(free.is_empty());
    }

    #[test]
    fn test_substitution_realigns_reordered_constraints() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let u = q("U", 1, vec![lit(1), Type::None]);
        let mut cases = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)]);
        cases.subst_mut_fn(&mut |x| (x == &t).then(|| tq(&u)));
        assert_eq!(cases, Type::quantified_cases(u, vec![lit(6), lit(5)]));
    }

    #[test]
    fn test_substitution_realigns_subset_constraints() {
        let t = q("T", 0, vec![Type::None, lit(1), lit(2)]);
        let u = q("U", 1, vec![lit(2), Type::None]);
        let mut cases = Type::quantified_cases(t.clone(), vec![lit(5), lit(6), lit(7)]);
        cases.subst_mut_fn(&mut |x| (x == &t).then(|| tq(&u)));
        assert_eq!(cases, Type::quantified_cases(u, vec![lit(7), lit(5)]));
    }

    #[test]
    fn test_substitution_erases_unmatched_or_ambiguous_constraints() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let u = q("U", 1, vec![lit(1), lit(3)]);
        let mut cases = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)]);
        cases.subst_mut_fn(&mut |x| (x == &t).then(|| tq(&u)));
        assert_eq!(cases, Type::union(vec![lit(5), lit(6)]));
        let d = q("D", 2, vec![lit(1), lit(1), Type::None]);
        let v = q("V", 3, vec![lit(1)]);
        let mut cases = Type::quantified_cases(d.clone(), vec![lit(5), lit(6), lit(7)]);
        cases.subst_mut_fn(&mut |x| (x == &d).then(|| tq(&v)));
        assert_eq!(cases, Type::union(vec![lit(5), lit(6), lit(7)]));
    }

    #[test]
    fn test_var_selector_defers_choice() {
        let uniques = UniqueFactory::new();
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let s = q("S", 1, vec![lit(1), Type::None]);
        let var = Type::Var(Var::new(&uniques));
        let ty = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)]);
        let mut deferred = ty.clone();
        deferred.subst_mut_fn(&mut |x| (x == &t).then(|| var.clone()));
        let Type::QuantifiedCases(x) = &deferred else {
            panic!("expected deferred cases")
        };
        assert_eq!(x.selector(), Some(&var));
        assert_eq!(x.resolve_selector(), None);
        // The quantified only supplies constraints once a variable selects the cases.
        let mut free = Vec::new();
        deferred.for_each_free_quantified(&mut |q| free.push(q.clone()));
        assert!(free.is_empty());
        assert_eq!(deferred.specialize_quantified(&t, 0), deferred);
        // Speculative subset checks snapshot the pending selector just like other variables.
        assert_eq!(
            deferred.collect_maybe_placeholder_vars(),
            var.collect_all_vars()
        );
        let mut unchanged = deferred.clone();
        unchanged.subst_mut_fn(&mut |x| (x == &t).then(|| lit(1)));
        assert_eq!(unchanged, deferred);
        let solve = |answer: Type| {
            let mut solved = deferred.clone();
            solved.transform_mut(&mut |x| {
                if x == &var {
                    *x = answer.clone();
                }
            });
            let Type::QuantifiedCases(x) = &solved else {
                panic!("expected deferred cases")
            };
            x.resolve_selector().expect("the selector is solved")
        };
        assert_eq!(solve(lit(1)), lit(6));
        assert_eq!(solve(Type::None), lit(5));
        assert_eq!(
            solve(tq(&s)),
            Type::quantified_cases(s.clone(), vec![lit(6), lit(5)])
        );
        assert_eq!(
            solve(Type::any_implicit()),
            Type::union(vec![lit(5), lit(6)])
        );
        assert!(deferred.as_quantified_cases().is_none());
        let Type::QuantifiedCases(pending) = deferred.clone() else {
            panic!("expected deferred cases")
        };
        assert_eq!(
            pending.into_selected_parts(),
            Err(Type::union(vec![lit(5), lit(6)]))
        );
        // A selector solved to a free quantified but not yet normalized is still specialized.
        let mut unnormalized = deferred.clone();
        unnormalized.transform_mut(&mut |x| {
            if x == &var {
                *x = tq(&s);
            }
        });
        assert_eq!(unnormalized.specialize_quantified(&s, 1), lit(5));
        let mut free = Vec::new();
        unnormalized.for_each_free_quantified(&mut |q| free.push(q.clone()));
        assert_eq!(free, vec![s.clone()]);
        let mut substituted = unnormalized;
        substituted.subst_mut_fn(&mut |x| (x == &s).then(|| lit(1)));
        assert_eq!(substituted, lit(6));
    }

    #[test]
    fn test_var_selector_identity_cases_collapse_to_selector() {
        let uniques = UniqueFactory::new();
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let var = Type::Var(Var::new(&uniques));
        let Type::QuantifiedCases(cases) = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)])
        else {
            panic!("expected cases")
        };
        let mut deferred = cases.with_cases(vec![lit(5), lit(6)]);
        deferred.subst_mut_fn(&mut |x| (x == &t).then(|| var.clone()));
        let Type::QuantifiedCases(cases) = deferred else {
            panic!("expected deferred cases")
        };
        assert_eq!(cases.with_cases(vec![Type::None, lit(1)]), var);
    }

    #[test]
    fn test_same_quantified_alignment() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let inner = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)]);
        let outer =
            Type::quantified_cases(t.clone(), vec![Type::Type(Box::new(inner.clone())), inner]);
        let Type::QuantifiedCases(x) = &outer else {
            panic!("expected cases")
        };
        assert_eq!(x.cases(), &[Type::Type(Box::new(lit(5))), lit(6)]);
    }

    #[test]
    fn test_independent_quantifieds_nest() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let u = q("U", 1, vec![Type::None, lit(1)]);
        let inner = Type::quantified_cases(u.clone(), vec![lit(5), lit(6)]);
        let outer = Type::quantified_cases(t.clone(), vec![inner.clone(), lit(7)]);
        assert_eq!(outer.specialize_quantified(&t, 0), inner);
        assert_eq!(
            outer.specialize_quantified(&u, 1),
            Type::quantified_cases(t, vec![lit(6), lit(7)])
        );
    }

    #[test]
    fn test_specialize_substitutes_free_quantified() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let ty = Type::quantified_cases(t.clone(), vec![Type::Type(Box::new(tq(&t))), lit(9)]);
        // The constructor already specialized the free `T` in case 0.
        assert_eq!(
            ty.specialize_quantified(&t, 0),
            Type::Type(Box::new(Type::None))
        );
    }

    #[test]
    fn test_subst() {
        let t = q("T", 0, vec![Type::None, lit(1)]);
        let s = q("S", 1, vec![Type::None, lit(1)]);
        let ty = Type::quantified_cases(t.clone(), vec![lit(5), lit(6)]);
        let mut renamed = ty.clone();
        renamed.subst_mut_fn(&mut |x| (x == &t).then(|| tq(&s)));
        assert_eq!(renamed, Type::quantified_cases(s, vec![lit(5), lit(6)]));
        let mut chosen = ty.clone();
        chosen.subst_mut_fn(&mut |x| (x == &t).then(|| lit(1)));
        assert_eq!(chosen, lit(6));
        let mut other = ty;
        other.subst_mut_fn(&mut |x| (x == &t).then(|| lit(3)));
        assert_eq!(other, Type::union(vec![lit(5), lit(6)]));
    }

    #[test]
    fn test_leaf_budget_erases() {
        let cs: Vec<Type> = (0..8).map(lit).collect();
        let t = q("T", 0, cs.clone());
        let u = q("U", 1, cs.clone());
        let v = q("V", 2, cs);
        let inner = Type::quantified_cases(u, (10..18).map(lit).collect());
        let mid = Type::quantified_cases(t, (0..8).map(|_| inner.clone()).collect());
        // All cases equal, so this collapsed to `inner`.
        assert_eq!(mid, inner);
        let w = q("W", 3, (0..8).map(lit).collect());
        let mid = Type::quantified_cases(
            w,
            (0..8)
                .map(|i| Type::concrete_tuple(vec![inner.clone(), lit(100 + i)]))
                .collect(),
        );
        assert!(matches!(mid, Type::QuantifiedCases(_)));
        let outer = Type::quantified_cases(
            v,
            (0..8)
                .map(|i| Type::Tuple(Tuple::Concrete(vec![mid.clone(), lit(100 + i)])))
                .collect(),
        );
        assert!(!matches!(outer, Type::QuantifiedCases(_)));
    }

    #[test]
    fn test_leaf_budget_counts_independent_type_arguments() {
        let t = q("T", 0, vec![lit(0), lit(1)]);
        let u = q("U", 1, (0..8).map(lit).collect());
        let v = q("V", 2, (0..8).map(lit).collect());
        let left = Type::quantified_cases(u, (10..18).map(lit).collect());
        let right = Type::quantified_cases(v, (20..28).map(lit).collect());
        let result = Type::quantified_cases(
            t,
            (0..2)
                .map(|i| Type::concrete_tuple(vec![left.clone(), right.clone(), lit(i)]))
                .collect(),
        );
        assert!(!matches!(result, Type::QuantifiedCases(_)));
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::max;
use std::ptr::eq as ptr_eq;

use itertools::Either;
use itertools::Itertools;
use pyrefly_types::callable::ArgCount;
use pyrefly_types::callable::ArgCounts;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamOverlay;
use pyrefly_types::dimension::ShapeError;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_types::meta_shape_dsl::ShapeTransform;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_output::DisplayOutput;
use pyrefly_types::type_output::TypeOutput;
use pyrefly_types::types::TArgs;
use pyrefly_util::display::Fmt;
use pyrefly_util::display::count;
use pyrefly_util::gas::Gas;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::VecExt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers::OverloadTrace;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::TargetWithTParams;
use crate::alt::callable::ArgMap;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::callable::CallWithTypes;
use crate::alt::callable::ReturnTypeResolutionError;
use crate::alt::expr::TypeOrExpr;
use crate::alt::unwrap::HintRef;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::solver::solver::ArgumentKey;
use crate::solver::solver::OverloadTable;
use crate::solver::solver::TypeVarSpecializationError;
use crate::types::callable::Callable;
use crate::types::callable::Params;
use crate::types::function::FuncMetadata;
use crate::types::function::Function;
use crate::types::literal::Lit;
use crate::types::quantified::Quantified;
use crate::types::type_var::Restriction;
use crate::types::types::Type;
use crate::types::types::Var;

struct CalledOverload<'f> {
    func: &'f TargetWithTParams<Function>,
    res: Type,
    ctor_targs: Option<TArgs>,
    table: OverloadTable,
    arg_errors: ErrorCollector,
    call_errors: ErrorCollector,
    specialization_errors: Vec<TypeVarSpecializationError>,
    return_type_errors: Vec<ReturnTypeResolutionError>,
    defaults_used: SmallSet<Quantified>,
    /// Maps each argument's position to the parameter it was matched against.
    argmap: ArgMap,
}

impl CalledOverload<'_> {
    fn num_match_errors(&self) -> usize {
        self.call_errors.len_hard() + self.specialization_errors.len()
    }

    fn has_match_errors(&self) -> bool {
        self.call_errors.has_hard() || !self.specialization_errors.is_empty()
    }

    fn has_hard_call_errors(&self) -> bool {
        self.call_errors.has_hard()
    }
}

/// Specializes the free occurrences of `q` in the signature of `target` to its `index`-th
/// constraint. A quantified bound by the target's own type parameters is distinct and left alone.
fn specialize_target(
    mut target: TargetWithTParams<Function>,
    q: &Quantified,
    index: usize,
) -> TargetWithTParams<Function> {
    if target
        .0
        .as_ref()
        .is_some_and(|ts| ts.iter().any(|t| t == q))
    {
        return target;
    }
    target.1.signature =
        match Type::Callable(Box::new(target.1.signature)).specialize_quantified(q, index) {
            Type::Callable(c) => *c,
            _ => unreachable!("specialization only replaces nested type variables"),
        };
    target
}

/// The choice that produced one expanded argument list at one expansion step: the index of a
/// union member, or the index of a declared constraint of a constrained quantified.
#[derive(Clone, Debug)]
pub struct ExpansionChoice {
    quantified: Option<Quantified>,
    index: usize,
}

/// An argument list produced by argument type expansion, together with the choices made at each
/// expansion step, in the order in which the steps were made.
#[derive(Clone, Debug)]
pub struct ExpandedArgs<'a> {
    pub args: Vec<CallArg<'a>>,
    pub keywords: Vec<CallKeyword<'a>>,
    pub choices: Vec<ExpansionChoice>,
}

impl ExpandedArgs<'_> {
    fn has_quantified_choices(&self) -> bool {
        self.choices.iter().any(|c| c.quantified.is_some())
    }

    /// Specializes the free occurrences of chosen quantifieds in the signature of `target`.
    fn specialize_target(
        &self,
        target: &TargetWithTParams<Function>,
    ) -> TargetWithTParams<Function> {
        self.choices
            .iter()
            .fold(target.clone(), |target, choice| match &choice.quantified {
                Some(q) => specialize_target(target, q, choice.index),
                None => target,
            })
    }

    /// Specializes `ty` to the constraints chosen for quantifieds while producing this list.
    fn specialize(&self, ty: Type) -> Type {
        self.choices
            .iter()
            .fold(ty, |ty, choice| match &choice.quantified {
                Some(q) => ty.specialize_quantified(q, choice.index),
                None => ty,
            })
    }
}

/// Performs argument type expansion for arguments to an overloaded function.
pub struct ArgsExpander<'a, Ans: LookupAnswer> {
    /// The index of the next argument to expand. Left is positional args; right, keyword args.
    idx: Either<usize, usize>,
    /// Current argument lists. Lists are in expansion order: the choice of the most recent step
    /// varies fastest, so lists sharing a prefix of choices are contiguous.
    arg_lists: Vec<ExpandedArgs<'a>>,
    /// Hard-coded limit to how many times we'll expand.
    gas: Gas,
    solver: &'a AnswersSolver<'a, 'a, Ans>,
}

impl<'a, Ans: LookupAnswer> ArgsExpander<'a, Ans> {
    const GAS: usize = 100;

    pub fn new(
        posargs: Vec<CallArg<'a>>,
        keywords: Vec<CallKeyword<'a>>,
        solver: &'a AnswersSolver<'a, 'a, Ans>,
    ) -> Self {
        Self {
            idx: if posargs.is_empty() {
                Either::Right(0)
            } else {
                Either::Left(0)
            },
            arg_lists: vec![ExpandedArgs {
                args: posargs,
                keywords,
                choices: Vec::new(),
            }],
            gas: Gas::new(Self::GAS as isize),
            solver,
        }
    }

    /// Expand the next argument and return the expanded argument lists.
    pub fn expand(
        &mut self,
        errors: &ErrorCollector,
        owner: &'a Owner<Type>,
    ) -> Option<Vec<(Vec<CallArg<'a>>, Vec<CallKeyword<'a>>)>> {
        Some(
            self.expand_impl(errors, owner, None)?
                .into_map(|x| (x.args, x.keywords)),
        )
    }

    /// Like `expand`, but an argument whose type is a constrained quantified `q`, or depends on
    /// one, is expanded by specializing `q` to each declared constraint in every argument of the
    /// list. Occurrences of `q` in other arguments therefore stay consistent with the choice, and
    /// the recorded choices identify `q` so that results can be reassembled into cases over `q`.
    pub fn expand_correlated(
        &mut self,
        errors: &ErrorCollector,
        owner: &'a Owner<Type>,
        call: &'a CallWithTypes,
    ) -> Option<Vec<ExpandedArgs<'a>>> {
        self.expand_impl(errors, owner, Some(call))
    }

    fn expand_impl(
        &mut self,
        errors: &ErrorCollector,
        owner: &'a Owner<Type>,
        call: Option<&'a CallWithTypes>,
    ) -> Option<Vec<ExpandedArgs<'a>>> {
        let idx = self.idx;
        let first = self.arg_lists.first()?;
        let (posargs_len, keywords_len) = (first.args.len(), first.keywords.len());
        // Determine the idx of the value we will try next if needed.
        self.idx = match idx {
            Either::Left(i) if i < posargs_len - 1 => Either::Left(i + 1),
            Either::Left(_) => Either::Right(0),
            Either::Right(i) if i < keywords_len => Either::Right(i + 1),
            Either::Right(_) => return None,
        };
        // Earlier correlated expansions specialize whole lists, so the same argument can have a
        // different type in each list and must be expanded separately for each one.
        let expansions = self
            .arg_lists
            .iter()
            .map(|list| {
                let value = match idx {
                    Either::Left(i) => match &list.args[i] {
                        CallArg::Arg(value) | CallArg::Star(value, ..) => value,
                    },
                    Either::Right(i) => &list.keywords[i].value,
                };
                let ty = value.infer(self.solver, errors);
                match (call, ty.as_quantified_cases()) {
                    (Some(_), Some((q, cases))) => Either::Right((q.clone(), cases.len())),
                    _ => Either::Left(self.expand_type(ty).into_map(|t| owner.push(t))),
                }
            })
            .collect::<Vec<_>>();
        let width = |expansion: &Either<Vec<&'a Type>, (Quantified, usize)>| match expansion {
            Either::Left(types) => types.len(),
            Either::Right((_, n)) => *n,
        };
        if expansions.iter().all(|e| width(e) == 0) {
            // Nothing to expand here, try the next argument.
            return self.expand_impl(errors, owner, call);
        }
        let mut new_arg_lists = Vec::new();
        for (list, expansion) in self.arg_lists.iter().zip(&expansions) {
            // A list whose argument does not expand is kept as a single choice, so that every
            // list records a choice at every step.
            for index in 0..max(width(expansion), 1) {
                let mut choices = list.choices.clone();
                let new_list = match expansion {
                    Either::Left(types) => {
                        choices.push(ExpansionChoice {
                            quantified: None,
                            index,
                        });
                        let mut new_list = ExpandedArgs {
                            args: list.args.clone(),
                            keywords: list.keywords.clone(),
                            choices,
                        };
                        if let Some(ty) = types.get(index) {
                            match idx {
                                Either::Left(i) => {
                                    let new_value = TypeOrExpr::Type(ty, list.args[i].range());
                                    new_list.args[i] = match list.args[i] {
                                        CallArg::Arg(_) => CallArg::Arg(new_value),
                                        CallArg::Star(_, range) => CallArg::Star(new_value, range),
                                    }
                                }
                                Either::Right(i) => {
                                    new_list.keywords[i].value =
                                        TypeOrExpr::Type(ty, list.keywords[i].range());
                                }
                            }
                        }
                        new_list
                    }
                    Either::Right((q, _)) => {
                        let call = call.expect("correlated expansion requires `CallWithTypes`");
                        choices.push(ExpansionChoice {
                            quantified: Some(q.clone()),
                            index,
                        });
                        ExpandedArgs {
                            args: call.specialized_vec_call_arg(
                                &list.args,
                                q,
                                index,
                                self.solver,
                                errors,
                            ),
                            keywords: call.specialized_vec_call_keyword(
                                &list.keywords,
                                q,
                                index,
                                self.solver,
                                errors,
                            ),
                            choices,
                        }
                    }
                };
                new_arg_lists.push(new_list);
                if self.gas.stop() {
                    // We've hit our hard-coded limit; stop expanding, and move `idx` past the
                    // end of the keywords so that subsequent `expand` calls know we're done.
                    self.idx = Either::Right(keywords_len);
                    return None;
                }
            }
        }
        self.arg_lists = new_arg_lists.clone();
        Some(new_arg_lists)
    }

    /// Expands a type according to https://typing.python.org/en/latest/spec/overload.html#argument-type-expansion.
    fn expand_type(&self, ty: Type) -> Vec<Type> {
        match ty {
            Type::Union(f) => f.members,
            Type::ClassType(cls) if cls.is_builtin("bool") => {
                vec![
                    Lit::Bool(true).to_implicit_type(),
                    Lit::Bool(false).to_implicit_type(),
                ]
            }
            Type::ClassType(cls)
                if self
                    .solver
                    .get_metadata_for_class(cls.class_object())
                    .is_enum() =>
            {
                self.solver
                    .get_enum_members(cls.class_object())
                    .into_iter()
                    .map(Lit::to_implicit_type)
                    .collect()
            }
            Type::Type(f) if matches!(&*f, Type::Union(_)) => {
                // Repeated match because pattern guards cannot move out of bindings.
                let Type::Union(u) = *f else {
                    unreachable!("guarded by matches! above")
                };
                u.members.into_map(|t| self.solver.heap.mk_type_of(t))
            }
            Type::Tuple(Tuple::Concrete(elements)) => {
                let mut count: usize = 1;
                let mut changed = false;
                let mut element_expansions = Vec::new();
                for e in elements {
                    let element_expansion = self.expand_type(e.clone());
                    if element_expansion.is_empty() {
                        element_expansions.push(vec![e].into_iter());
                    } else {
                        let len = element_expansion.len();
                        count = count.saturating_mul(len);
                        if count > Self::GAS {
                            return Vec::new();
                        }
                        changed = true;
                        element_expansions.push(element_expansion.into_iter());
                    }
                }
                // Enforce a hard-coded limit on the number of expansions for perf reasons.
                if count <= Self::GAS && changed {
                    element_expansions
                        .into_iter()
                        .multi_cartesian_product()
                        .map(|x| self.solver.heap.mk_concrete_tuple(x))
                        .collect()
                } else {
                    Vec::new()
                }
            }

            // a constraind typevar argument is one of its constraints, so expand like a union ie try each constraint against overloads + union the matched returns
            Type::Quantified(q) => match q.restriction() {
                Restriction::Constraints(constraints) => constraints.clone(),
                _ => Vec::new(),
            },

            _ => Vec::new(),
        }
    }
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Finish a return type against this call's solutions.
    pub(crate) fn finish_return(
        &self,
        overload_table: &OverloadTable,
        t: Type,
    ) -> (Type, Vec<ShapeError>) {
        let per_row = self.solver().per_row(overload_table, || {
            self.solver().for_return_boundary(t.clone())
        });
        let type_level_dsl_errors = if per_row.len() == 1 {
            per_row.first().1.clone()
        } else {
            Vec::new()
        };
        let per_row = per_row.mapped(|(result, _)| result);
        (
            self.combine_overload_results(per_row, overload_table),
            type_level_dsl_errors,
        )
    }

    /// Fold one result per overload table row into a single type.
    pub(crate) fn combine_overload_results(
        &self,
        per_row: Vec1<Type>,
        overload_table: &OverloadTable,
    ) -> Type {
        let first = per_row.first();
        if per_row.iter().skip(1).all(|other| other == first) {
            return first.clone();
        }
        if overload_table.is_ambiguous() {
            return match self.disambiguate_overload_results(&per_row) {
                Some(index) => per_row[index].clone(),
                None => self.heap.mk_any_implicit(),
            };
        }
        Type::combine_overload_results(per_row.into_vec(), self.heap)
            .expect("a nonempty collection of results can always be combined")
    }

    /// Reassembles the results of calls with expanded argument lists, where `results[i]` is the
    /// result for `lists[i]` and every list shares the same first `depth` choices. Lists with the
    /// same prefix made the same choice kind at each step, because they expanded the same type. Results that
    /// differ in the choice of a union member are unioned. Results that differ in the constraint
    /// chosen for a quantified become cases over that quantified, preserving which constraint
    /// produced which result.
    fn combine_expanded_results(
        &self,
        lists: &[ExpandedArgs],
        mut results: Vec<Type>,
        depth: usize,
    ) -> Type {
        let Some(choice) = lists[0].choices.get(depth) else {
            assert_eq!(results.len(), 1, "lists with identical choices are unique");
            return results.pop().expect("there is one result");
        };
        // Lists sharing a prefix of choices are contiguous, and their choices at `depth` are in
        // ascending order, so each run of equal indices is one group.
        let mut groups = Vec::new();
        let mut results = results.into_iter();
        for group in lists.chunk_by(|x, y| x.choices[depth].index == y.choices[depth].index) {
            groups.push(self.combine_expanded_results(
                group,
                results.by_ref().take(group.len()).collect(),
                depth + 1,
            ));
        }
        match &choice.quantified {
            Some(q) => Type::quantified_cases(q.clone(), groups),
            None => self.unions(groups),
        }
    }

    /// The quantified of the first live `QuantifiedCases` in the argument types, provided that
    /// evaluating the call once per combination of the constraints of every such quantified stays
    /// within the argument type expansion limit. Pending cases have no live quantified.
    fn live_case_quantified(
        &self,
        args: &[CallArg],
        keywords: &[CallKeyword],
    ) -> Option<Quantified> {
        let mut quantifieds: Vec<&Quantified> = Vec::new();
        let values = args
            .iter()
            .map(|arg| match arg {
                CallArg::Arg(value) | CallArg::Star(value, _) => value,
            })
            .chain(keywords.iter().map(|kw| &kw.value));
        for value in values {
            if let TypeOrExpr::Type(ty, _) = value {
                let mut free = Vec::new();
                ty.for_each_free_quantified(&mut |q| free.push(q));
                ty.universe(&mut |t| {
                    if let Type::QuantifiedCases(cases) = t
                        && let Some(q) = cases.quantified()
                        && free.contains(&q)
                        && !quantifieds.contains(&q)
                    {
                        quantifieds.push(q);
                    }
                });
            }
        }
        let combinations = quantifieds
            .iter()
            .try_fold(1usize, |n, q| match q.restriction() {
                Restriction::Constraints(cs) => n.checked_mul(cs.len()),
                _ => unreachable!("cases are indexed by a constrained type variable"),
            })?;
        if combinations > ArgsExpander::<Ans>::GAS {
            return None;
        }
        quantifieds.first().map(|q| (*q).clone())
    }

    /// Calls an overloaded function, returning the return type, the closest matching overload
    /// signature, and the solutions that signature settled on.
    pub fn call_overloads(
        &self,
        overloads: Vec1<TargetWithTParams<Function>>,
        metadata: &FuncMetadata,
        shape_transform: Option<&ShapeTransform>,
        self_obj: Option<Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        errors: &ErrorCollector,
        return_errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        hint: Option<HintRef>,
        // If we're constructing a class, its type arguments. A successful call will fill these in.
        ctor_targs: Option<&mut TArgs>,
    ) -> (Type, Callable, OverloadTable) {
        // There may be Expr values in args and keywords.
        // If we infer them for each overload, we may end up inferring them multiple times.
        // If those overloads contain nested overloads, then we can easily end up with O(2^n) perf.
        // Therefore, flatten all TypeOrExpr's into Type before we start
        let call = CallWithTypes::new();
        let args = call.vec_call_arg(args, self, errors);
        let keywords = call.vec_call_keyword(keywords, self, errors);

        // An argument whose value depends on the constraint chosen for a live quantified is
        // evaluated once per constraint, even if the unsplit call would match. Otherwise a generic
        // overload would solve its type variables to the union of the cases, losing which
        // constraint produced which result. Specializing removes the quantified, so each nested
        // call splits on fewer quantifieds; the total number of calls is bounded up front.
        // Constructors share mutable type arguments across __new__/__init__, so they use the
        // ordinary inference path rather than committing different instantiations per case.
        if ctor_targs.is_none()
            && let Some(q) = self.live_case_quantified(&args, &keywords)
        {
            let Restriction::Constraints(constraints) = q.restriction() else {
                unreachable!("cases are indexed by a constrained type variable")
            };
            let mut signature = None;
            let case_errors = self.error_collector();
            let results = (0..constraints.len())
                .map(|index| {
                    let case_call = CallWithTypes::new();
                    let hint_ty = hint
                        .map(|hint| hint.map_types(self, |ty| ty.specialize_quantified(&q, index)));
                    let (ty, case_signature, _) = self.call_overloads(
                        overloads.mapped_ref(|t| specialize_target(t.clone(), &q, index)),
                        metadata,
                        shape_transform,
                        self_obj
                            .as_ref()
                            .map(|t| t.specialize_quantified(&q, index)),
                        &case_call.specialized_vec_call_arg(&args, &q, index, self, &case_errors),
                        &case_call.specialized_vec_call_keyword(
                            &keywords,
                            &q,
                            index,
                            self,
                            &case_errors,
                        ),
                        arguments_range,
                        &case_errors,
                        return_errors,
                        context,
                        HintRef::with_ty_opt(hint, hint_ty.as_ref()),
                        None,
                    );
                    signature.get_or_insert(case_signature);
                    ty
                })
                .collect();
            errors.extend_case_errors(case_errors, |_| {
                format!("Call is invalid for some constraints of `{q}`")
            });
            return (
                Type::quantified_cases(q, results),
                signature.expect("a constrained type variable has constraints"),
                // Constraint choices are universal, not alternative overload solutions. As for
                // union callees, do not merge the individual calls' overload tables.
                OverloadTable::default(),
            );
        }

        // Evaluate the call following https://typing.python.org/en/latest/spec/overload.html#overload-call-evaluation.

        // Overloads specialized during argument type expansion, which must outlive the results
        // that refer to them.
        let specialized_targets = Owner::new();

        // Step 1: eliminate overloads that accept an incompatible number of arguments.
        let mut arity_closest_overload = None;
        let arity_compatible_overloads = overloads
            .iter()
            .filter(|overload| {
                let arg_counts = overload.1.signature.arg_counts();
                let mismatch_size =
                    self.arity_mismatch_size(&arg_counts, self_obj.as_ref(), &args, &keywords);
                if arity_closest_overload
                    .as_ref()
                    .is_none_or(|(_, n)| *n > mismatch_size)
                {
                    arity_closest_overload = Some((*overload, mismatch_size));
                }
                mismatch_size == 0
            })
            .collect::<Vec<_>>();
        let (closest_overload, matched) = match Vec1::try_from_vec(arity_compatible_overloads) {
            Err(_) => (
                CalledOverload {
                    func: arity_closest_overload.unwrap().0,
                    res: self.heap.mk_any_error(),
                    ctor_targs: None,
                    table: OverloadTable::default(),
                    arg_errors: self.error_collector(),
                    call_errors: self.error_collector(),
                    specialization_errors: Vec::new(),
                    return_type_errors: Vec::new(),
                    defaults_used: SmallSet::new(),
                    argmap: ArgMap::new(),
                },
                false,
            ),
            Ok(arity_compatible_overloads) => {
                // Step 2: evaluate each overload as a regular (non-overloaded) call.
                // Note: steps 4-6 are performed in `find_closest_overload`.
                let (mut closest_overload, mut matched) = self.find_closest_overload(
                    &arity_compatible_overloads,
                    metadata,
                    shape_transform,
                    self_obj.as_ref(),
                    &args,
                    &keywords,
                    arguments_range,
                    errors,
                    hint,
                    &ctor_targs,
                );

                // Step 3: argument type expansion. When the mypy-compatibility flag is on, we also
                // use it to narrow an already-matched call to a more precise return type.
                // Unlike the up-front split of dependent values, this is a fallback for a failed
                // call, chiefly on plain constrained TypeVars and ordinary unions. Live cases
                // only remain here when construction or the combination budget prevented the
                // up-front split; the expander applies its own budget before evaluating calls.
                let refine = matched
                    && self.solver().config.legacy_overload_expansion
                    && matches!(&closest_overload.res, Type::Union(_));
                let mut args_expander = ArgsExpander::new(args.clone(), keywords.clone(), self);
                let owner = Owner::new();
                let expansion_call = CallWithTypes::new();
                'outer: while (!matched || refine)
                    && let Some(arg_lists) =
                        args_expander.expand_correlated(errors, &owner, &expansion_call)
                {
                    // Expand by one argument (for example, try splitting up union types), and try the call with each
                    // resulting arguments list.
                    // - If all expanded lists match, we union all return types together and declare a successful match
                    // - If any do not match, we move on to the next splittable argument (if we run out of args to split,
                    //   we'll wind up with a failed match and our best guess at the correct overload)
                    let mut matched_overloads = Vec::new();
                    for list in arg_lists.iter() {
                        let hint_ty = hint
                            .filter(|_| list.has_quantified_choices())
                            .map(|hint| hint.map_types(self, |ty| list.specialize(ty.clone())));
                        let hint = match &hint_ty {
                            Some(ty) => HintRef::with_ty_opt(hint, Some(ty)),
                            None => hint,
                        };
                        // Free occurrences of a split quantified in the signatures and the
                        // receiver must agree with the constraint chosen for the arguments.
                        let specialized = list.has_quantified_choices().then(|| {
                            specialized_targets.push(
                                arity_compatible_overloads
                                    .mapped_ref(|t| list.specialize_target(t)),
                            )
                        });
                        let specialized_refs = specialized.as_ref().map(|ts| ts.mapped_ref(|t| t));
                        let cur_self_obj = self_obj.clone().map(|t| list.specialize(t));
                        let (mut cur_closest, cur_matched) = self.find_closest_overload(
                            specialized_refs
                                .as_ref()
                                .unwrap_or(&arity_compatible_overloads),
                            metadata,
                            shape_transform,
                            cur_self_obj.as_ref(),
                            &list.args,
                            &list.keywords,
                            arguments_range,
                            errors,
                            hint,
                            &ctor_targs,
                        );
                        if !cur_matched {
                            continue 'outer;
                        }
                        if let Some(specialized) = specialized {
                            let index = specialized
                                .iter()
                                .position(|t| ptr_eq(t, cur_closest.func))
                                .expect("the closest overload is one of the specialized overloads");
                            cur_closest.func = arity_compatible_overloads[index];
                        }
                        matched_overloads.push(cur_closest);
                    }
                    if !matched_overloads.is_empty() {
                        if matched {
                            // Adopt only when an expanded member hit a `Never` overload, dropping it
                            // from the union; otherwise try the next argument rather than give up.
                            let dropped_arm = matched_overloads.iter().any(|o| o.res.is_never());
                            let expanded_res = self.unions(matched_overloads.into_map(|o| o.res));
                            if dropped_arm
                                && self.is_subset_eq(&expanded_res, &closest_overload.res)
                                && !self.is_equivalent(&expanded_res, &closest_overload.res)
                            {
                                closest_overload.res = expanded_res;
                                break;
                            }
                            continue 'outer;
                        }
                        let first_overload = &matched_overloads[0];
                        let func = first_overload.func;
                        let ctor_targs = first_overload.ctor_targs.clone();
                        // Several signatures matched and their results are unioned, so no single
                        // table of solutions describes the call any more.
                        let table = OverloadTable::default();
                        let argmap = first_overload.argmap.clone();
                        let arg_errors = self.error_collector();
                        let specialization_errors = first_overload.specialization_errors.clone();
                        let return_type_errors = matched_overloads
                            .iter()
                            .flat_map(|overload| overload.return_type_errors.iter().cloned())
                            .unique()
                            .collect();
                        let defaults_used = matched_overloads
                            .iter()
                            .flat_map(|overload| overload.defaults_used.iter().cloned())
                            .collect();
                        closest_overload = CalledOverload {
                            func,
                            ctor_targs,
                            table,
                            argmap,
                            res: self.combine_expanded_results(
                                &arg_lists,
                                matched_overloads.into_map(|o| {
                                    arg_errors.extend(o.arg_errors);
                                    o.res
                                }),
                                0,
                            ),
                            arg_errors,
                            call_errors: self.error_collector(),
                            specialization_errors,
                            return_type_errors,
                            defaults_used,
                        };
                        matched = true;
                        break;
                    }
                }
                (
                    closest_overload,
                    // If there was only one overload with the right arity, it definitely matched.
                    matched || arity_compatible_overloads.len() == 1,
                )
            }
        };

        if matched
            && let Some(targs) = ctor_targs
            && let Some(chosen_targs) = closest_overload.ctor_targs
        {
            *targs = chosen_targs;
        }
        // Record the closest overload to power IDE services. Guard on the trace
        // sink before building the (per-signature cloned) traces, which would
        // otherwise be wasted work in a normal non-tracing check.
        if self.current().tracing_enabled() {
            let mut overload_trace = |target: &TargetWithTParams<Function>| {
                let tparams = target
                    .0
                    .as_ref()
                    .filter(|tparams| !tparams.is_empty())
                    .cloned();
                OverloadTrace::new(target.1.signature.clone(), tparams)
            };
            let all_overload_traces = overloads.iter().map(&mut overload_trace).collect();
            let closest_overload_trace = overload_trace(closest_overload.func);
            self.record_overload_trace(
                arguments_range,
                all_overload_traces,
                closest_overload_trace,
                matched,
            );
        }
        if matched {
            // If the selected overload is deprecated, we log a deprecation error.
            if let Some(deprecation) = &closest_overload.func.1.metadata.flags.deprecation {
                let header = format!(
                    "Call to deprecated overload `{}`",
                    closest_overload
                        .func
                        .1
                        .metadata
                        .kind
                        .format(self.module().name())
                );
                let detail = deprecation.as_error_detail();
                let mut error_builder =
                    errors.error_builder(arguments_range, ErrorKind::Deprecated, header);
                if let Some(detail) = detail {
                    error_builder = error_builder.with_detail(detail);
                }
                error_builder.with_context(context).emit();
            }
            errors.extend(closest_overload.arg_errors);
            errors.extend(closest_overload.call_errors);
            if let Ok(specialization_errors) =
                Vec1::try_from_vec(closest_overload.specialization_errors)
            {
                self.add_specialization_errors(
                    specialization_errors,
                    arguments_range,
                    errors,
                    None,
                );
            }
            self.add_return_type_resolution_errors(
                closest_overload.return_type_errors,
                arguments_range,
                return_errors,
                None,
            );
            (
                closest_overload.res,
                closest_overload.func.1.signature.clone(),
                closest_overload.table,
            )
        } else {
            if let Ok(specialization_errors) =
                Vec1::try_from_vec(closest_overload.specialization_errors)
            {
                self.add_specialization_errors(
                    specialization_errors,
                    arguments_range,
                    &closest_overload.call_errors,
                    None,
                );
            }
            self.overload_error(
                &overloads,
                metadata,
                self_obj.as_ref(),
                &args,
                &keywords,
                arguments_range,
                errors,
                context,
                &closest_overload.func.1.signature,
                closest_overload.call_errors,
                closest_overload.argmap,
            );
            (
                self.heap.mk_any_error(),
                closest_overload.func.1.signature.clone(),
                OverloadTable::default(),
            )
        }
    }

    /// Read and combine the branches of an overloaded value, dropping branches that don't apply.
    pub fn read_overloaded_branches(
        &self,
        branches: &Vec1<Type>,
        errors: &ErrorCollector,
        read: &dyn Fn(&Type, &ErrorCollector) -> Type,
    ) -> Type {
        let mut accepted = Vec::with_capacity(branches.len());
        let mut first_failure = None;
        for branch in branches {
            let attempt = self.error_collector();
            let result = read(branch, &attempt);
            if attempt.is_empty() {
                accepted.push(result);
            } else {
                first_failure.get_or_insert((result, attempt));
            }
        }
        // `combine_overload_results` answers `None` only for no results at all, which here means
        // no branch accepted the read.
        match Type::combine_overload_results(accepted, self.heap) {
            Some(combined) => combined,
            None => {
                let (result, attempt) = first_failure.expect("an overloaded type is never empty");
                errors.extend(attempt);
                result
            }
        }
    }

    fn arity_mismatch_size(
        &self,
        expected_arg_counts: &ArgCounts,
        self_obj: Option<&Type>,
        posargs: &[CallArg],
        keywords: &[CallKeyword],
    ) -> usize {
        // If the number of non-variadic args is less than the min or more than the max, get the
        // absolute difference between actual and expected. We ignore variadic args because we
        // can't figure out how many args they contribute without inferring their types, which we
        // want to avoid to keep this arity check lightweight.
        let (n_posargs, has_varargs) = {
            let n = posargs
                .iter()
                .filter(|arg| matches!(arg, CallArg::Arg(_)))
                .count();
            ((self_obj.is_some() as usize) + n, posargs.len() > n)
        };
        let n_keywords = keywords.iter().filter(|kw| kw.arg.is_some()).count();
        let has_kwargs = keywords.len() > n_keywords;
        let mismatch_size = |count: &ArgCount, n, variadic| {
            // Check for too few args.
            let min_mismatch = count
                .min
                .saturating_sub(if variadic { count.min } else { n });
            // Check for too many args.
            let max_mismatch = n.saturating_sub(count.max.unwrap_or(n));
            max(min_mismatch, max_mismatch)
        };
        let pos_mismatch = mismatch_size(&expected_arg_counts.positional, n_posargs, has_varargs);
        let kw_mismatch = mismatch_size(&expected_arg_counts.keyword, n_keywords, has_kwargs);
        let overall_mismatch = mismatch_size(
            &expected_arg_counts.overall,
            n_posargs + n_keywords,
            has_varargs || has_kwargs,
        );
        // overall_mismatch will double-count, but this is ok because all we care about is whether
        // the mismatch is 0 (correct arity) and relative mismatch sizes between overloads
        pos_mismatch + kw_mismatch + overall_mismatch
    }

    fn overload_error(
        &self,
        overloads: &[TargetWithTParams<Function>],
        metadata: &FuncMetadata,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
        closest_overload_signature: &Callable,
        closest_overload_call_errors: ErrorCollector,
        mut closest_overload_argmap: ArgMap,
    ) {
        // Build a string showing the argument types for error messages
        let mut arg_type_strs = Vec::new();
        for arg in args {
            let (ty, prefix) = match arg {
                CallArg::Arg(value) => (value.infer(self, errors), ""),
                CallArg::Star(value, _) => (value.infer(self, errors), "*"),
            };
            let ty_display = self.for_display(ty);
            arg_type_strs.push(format!("{}{}", prefix, ty_display));
        }
        for kw in keywords {
            let ty = kw.value.infer(self, errors);
            let ty_display = self.for_display(ty);
            if let Some(arg_name) = kw.arg {
                arg_type_strs.push(format!("{}={}", arg_name.as_str(), ty_display));
            } else {
                arg_type_strs.push(format!("**{}", ty_display));
            }
        }
        let args_display = format!("({})", arg_type_strs.join(", "));

        let header = format!(
            "No matching overload found for function `{}` called with arguments: {}",
            metadata.kind.format(self.module().name()),
            args_display
        );
        let mut details = vec!["Possible overloads:".to_owned()];
        // Build an overlay of relevant parameters. We'll show only these parameters when printing overloads.
        let self_offset = usize::from(self_obj.is_some());
        if self_obj.is_some() {
            // We strip `self` from the displayed signatures, so drop it from the overlay as well.
            closest_overload_argmap
                .arg_to_param
                .remove(&ArgumentKey::Positional(0));
        }
        let signature_overlay = {
            // If call errors is empty, the call failed due to an arity mismatch. Show all
            // parameters so the user can see the number of parameters in each signature.
            if closest_overload_call_errors.is_empty()
                // If any args are unpacked, we don't know for sure which parameters are matched,
                // so show all of them.
                || args.iter().any(|arg| matches!(arg, CallArg::Star(..)))
                || keywords.iter().any(|kw| kw.arg.is_none())
                // If an unexpected argument was passed in, show all the parameters to help the user
                // figure out which one they actually meant to match.
                || (0..args.len()).map(|i| ArgumentKey::Positional(self_offset + i)).chain((0..keywords.len()).map(ArgumentKey::Keyword)).any(|key| !closest_overload_argmap.arg_to_param.contains_key(&key))
            {
                ParamOverlay::All
            } else {
                // Show only parameters that were matched by passed arguments and required
                // parameters that were not matched.
                let names = closest_overload_argmap
                    .arg_to_param
                    .into_values()
                    .map(|p| p.name)
                    .chain(closest_overload_argmap.unmatched_params)
                    .collect::<Option<SmallSet<_>>>();
                match names {
                    Some(names) => ParamOverlay::Subset(names),
                    None => ParamOverlay::All,
                }
            }
        };
        for overload in overloads {
            let suffix = if overload.1.signature == *closest_overload_signature {
                " [closest match]"
            } else {
                ""
            };
            let signature = match self_obj {
                Some(_) => overload
                    .1
                    .signature
                    .strip_first_param()
                    .unwrap_or_else(|| overload.1.signature.clone()),
                None => overload.1.signature.clone(),
            };
            let type_for_display = self
                .solver()
                .for_display(self.heap.mk_callable_from(signature));
            let display_context = TypeDisplayContext::new(&[&type_for_display]);
            let signature_for_display = match &type_for_display {
                Type::Callable(callable) => callable,
                _ => unreachable!("Expected a callable"),
            };
            let signature_for_display = match &signature_for_display.params {
                Params::List(params) => {
                    if let ParamOverlay::Subset(names) = &signature_overlay
                        && let cur_names = params
                            .items()
                            .iter()
                            .filter_map(|p| p.name())
                            .collect::<SmallSet<_>>()
                        && names.iter().any(|name| !cur_names.contains(name))
                    {
                        // If any of the parameters we want to show don't exist in the current
                        // signature, be conservative and show everything.
                        format!("{type_for_display}")
                    } else {
                        format!(
                            "{}",
                            Fmt(|f| {
                                let mut output = DisplayOutput::new(&display_context, f);
                                let write_type = |t: &Type, o: &mut DisplayOutput| {
                                    display_context.fmt_helper_generic(t, false, o)
                                };
                                output.write_str("(")?;
                                params.fmt_with_type(
                                    &mut output,
                                    &write_type,
                                    &signature_overlay,
                                )?;
                                output.write_str(") -> ")?;
                                write_type(&signature_for_display.ret, &mut output)
                            })
                        )
                    }
                }
                _ => format!("{type_for_display}"),
            };
            details.push(format!("  {signature_for_display}{suffix}"));
        }
        let mut builder = errors
            .error_builder(arguments_range, ErrorKind::NoMatchingOverload, header)
            .with_context(context)
            .with_details(details);
        if closest_overload_call_errors.is_empty() {
            // If there were no call errors, the failure must have been an arity mismatch.
            let mut arg_counts = closest_overload_signature.arg_counts();
            let nposargs = args.len();
            let nkwargs = keywords.len();
            let missing_self_param = self_obj.is_some() && arg_counts.positional.max == Some(0);
            if self_obj.is_some() {
                arg_counts.positional.min = arg_counts.positional.min.saturating_sub(1);
                if let Some(max) = arg_counts.positional.max {
                    arg_counts.positional.max = Some(max.saturating_sub(1));
                }
                arg_counts.overall.min = arg_counts.overall.min.saturating_sub(1);
                if let Some(max) = arg_counts.overall.max {
                    arg_counts.overall.max = Some(max.saturating_sub(1));
                }
            }
            let check = |actual, expected: ArgCount, descriptor_prefix| {
                let descriptor = format!("{descriptor_prefix}argument");
                if actual < expected.min {
                    Some(format!(
                        "Expected at least {}, got {actual}",
                        count(expected.min, &descriptor)
                    ))
                } else if let Some(max) = expected.max
                    && actual > max
                {
                    Some(format!(
                        "Expected at most {}, got {actual}",
                        count(max, &descriptor)
                    ))
                } else {
                    None
                }
            };
            let arity_mismatch = if missing_self_param {
                "This method has no `self` parameter to receive the implicit instance argument. Add a `self` parameter.".to_owned()
            } else {
                check(nposargs + nkwargs, arg_counts.overall, "").unwrap_or_else(|| {
                    check(nposargs, arg_counts.positional, "positional ").unwrap_or_else(|| {
                        check(nkwargs, arg_counts.keyword, "keyword ")
                            .expect("Overload evaluation: expected arity mismatch not found")
                    })
                })
            };
            builder = builder.with_detail(arity_mismatch);
        } else {
            builder = builder.with_errors_as_details(closest_overload_call_errors);
        }
        builder.emit();
    }

    /// Returns the overload that matches the given arguments, or the one that produces the fewest
    /// errors if none matches, plus a bool to indicate whether we found a match.
    fn find_closest_overload<'c>(
        &self,
        overloads: &Vec1<&'c TargetWithTParams<Function>>,
        metadata: &FuncMetadata,
        shape_transform: Option<&ShapeTransform>,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        errors: &ErrorCollector,
        hint: Option<HintRef>,
        ctor_targs: &Option<&mut TArgs>,
    ) -> (CalledOverload<'c>, bool) {
        // Collect placeholder vars so we can save/restore them around each overload evaluation. This
        // prevents premature pinning of vars on failed overload calls.
        let placeholder_vars = self.collect_placeholder_vars(self_obj, args, keywords);

        let mut matched_overloads = Vec::with_capacity(overloads.len());
        let mut closest_unmatched_overload: Option<CalledOverload<'c>> = None;
        for callable in overloads {
            let snapshot = self.solver().snapshot_exact_vars(&placeholder_vars);
            let mut called_overload = self.call_overload(
                callable,
                metadata,
                shape_transform,
                self_obj,
                args,
                keywords,
                arguments_range,
                None, // don't use the hint yet, it shouldn't influence overload selection
                None,
                ctor_targs,
            );
            // Each overload's argmap should use its own var solutions.
            for var in placeholder_vars.iter() {
                self.solver().force_var(*var);
            }
            for param in called_overload.argmap.arg_to_param.values_mut() {
                self.solver().expand_mut(&mut param.ty);
            }
            self.solver().restore_vars(snapshot);
            let n_errors = called_overload.num_match_errors();
            if n_errors == 0 {
                matched_overloads.push(called_overload);
            } else {
                match &closest_unmatched_overload {
                    Some(overload) if overload.num_match_errors() <= n_errors => {}
                    _ => {
                        closest_unmatched_overload = Some(called_overload);
                    }
                }
            }
        }
        if matched_overloads.is_empty() {
            // There's always at least one overload, so if none of them matched, the closest overload must be non-None.
            (closest_unmatched_overload.unwrap(), false)
        } else {
            // If there are multiple overloads, use steps 4-6 here to select one:
            // https://typing.python.org/en/latest/spec/overload.html#overload-call-evaluation.
            let spec_compliant = self.solver().config.spec_compliant_overloads;
            if matched_overloads.len() > 1 {
                // Step 4: if any arguments supply an unknown number of args and at least one
                // overload has a corresponding variadic parameter, eliminate overloads without
                // this parameter.
                let nargs_unknown = args.iter().any(|arg| match arg {
                    CallArg::Arg(_) => false,
                    CallArg::Star(val, _) => {
                        !matches!(val.infer(self, errors), Type::Tuple(Tuple::Concrete(_)))
                    }
                });
                if nargs_unknown {
                    let has_varargs = |o: &CalledOverload<'_>| {
                        matches!(
                            &o.func.1.signature.params, Params::List(params)
                            if params.items().iter().any(|p| matches!(p, Param::Varargs(..))))
                    };
                    if matched_overloads.iter().any(has_varargs) {
                        matched_overloads.retain(has_varargs);
                    }
                }
                let nkeywords_unknown = keywords.iter().any(|kw| {
                    kw.arg.is_none() && !matches!(kw.value.infer(self, errors), Type::TypedDict(_))
                });
                if nkeywords_unknown {
                    let has_kwargs = |o: &CalledOverload<'_>| {
                        matches!(
                            &o.func.1.signature.params, Params::List(params)
                            if params.items().iter().any(|p| matches!(p, Param::Kwargs(..))))
                    };
                    if matched_overloads.iter().any(has_kwargs) {
                        matched_overloads.retain(has_kwargs);
                    }
                }
            }
            if matched_overloads.len() > 1 {
                // Step 5: for each overload, check whether it's the case that all possible
                // materializations of each argument are assignable to the corresponding parameter.
                // If so, eliminate all subsequent overloads.
                //
                // Additional filter (non-spec-compliant): only materialize arguments that have
                // multiple possible parameter types. If an argument contains `Any` but has the
                // same parameter type in all candidate overloads, it does not contribute to
                // ambiguity in overload selection. This matches pyright, mypy, and ty.
                let owner = Owner::new();
                let mut changed = false;
                let should_materialize = |position| {
                    if spec_compliant {
                        return true;
                    }
                    let mut param_types = matched_overloads
                        .iter()
                        .filter_map(|o| o.argmap.arg_to_param.get(&position).map(|p| &p.ty));
                    let Some(first) = param_types.next() else {
                        // If we can't find the expected type, be conservative and assume there may be multiple.
                        return true;
                    };
                    for t in param_types {
                        if !self.is_equivalent(first, t) {
                            return true;
                        }
                    }
                    false
                };
                let self_offset = usize::from(self_obj.is_some());
                let materialized_args = args
                    .iter()
                    .enumerate()
                    .map(|(i, arg)| {
                        let (materialized_arg, arg_changed) =
                            if should_materialize(ArgumentKey::Positional(self_offset + i)) {
                                arg.materialize(self, errors, &owner)
                            } else {
                                (arg.clone(), false)
                            };
                        changed |= arg_changed;
                        materialized_arg
                    })
                    .collect::<Vec<_>>();
                let materialized_keywords = keywords
                    .iter()
                    .enumerate()
                    .map(|(i, kw)| {
                        let (materialized_kw, kw_changed) =
                            if should_materialize(ArgumentKey::Keyword(i)) {
                                kw.materialize(self, errors, &owner)
                            } else {
                                (kw.clone(), false)
                            };
                        changed |= kw_changed;
                        materialized_kw
                    })
                    .collect::<Vec<_>>();
                let split_point = if !changed {
                    // Shortcut: if the arguments haven't changed, we know that the first overload
                    // matches and we can eliminate all the rest.
                    Some(1)
                } else {
                    matched_overloads
                        .iter()
                        .find_position(|o| {
                            let snapshot = self.solver().snapshot_exact_vars(&placeholder_vars);
                            let res = self.call_overload(
                                o.func,
                                metadata,
                                shape_transform,
                                self_obj,
                                &materialized_args,
                                &materialized_keywords,
                                arguments_range,
                                None, // don't use the hint yet, it shouldn't influence overload selection
                                None,
                                &None,
                            );
                            self.solver().restore_vars(snapshot);
                            !res.has_match_errors()
                        })
                        .map(|(split_point, _)| split_point + 1)
                };
                if let Some(split_point) = split_point {
                    let _ = matched_overloads.split_off(split_point);
                }
            }
            let selected_overload = self.disambiguate_overload_results(
                &matched_overloads
                    .iter()
                    .map(|o| o.res.clone())
                    .collect::<Vec<_>>(),
            );
            if let Some(idx) = selected_overload {
                let overload = matched_overloads
                    .into_iter()
                    .nth(idx)
                    .expect("Could not find selected overload");
                // Now that we've selected an overload, use the hint to contextually type the arguments.
                let contextual_overload = self.call_overload(
                    overload.func,
                    metadata,
                    shape_transform,
                    self_obj,
                    args,
                    keywords,
                    arguments_range,
                    hint,
                    Some(&overload.defaults_used),
                    ctor_targs,
                );
                (
                    // The contextual pass may legitimately introduce late resolution errors, so
                    // we only fall back to the no-hint version on hard call errors. See
                    // `test::generic_restriction::test_nested_call_of_overloaded_function_preserves_bound`.
                    if !contextual_overload.has_hard_call_errors() {
                        contextual_overload
                    } else {
                        overload
                    },
                    true,
                )
            } else {
                // Ambiguous call, return Any. Arbitrarily use the first overload as the matched one.
                let first_overload = matched_overloads
                    .into_iter()
                    .next()
                    .expect("Expected at least one overload");
                (
                    CalledOverload {
                        res: self.heap.mk_any_implicit(),
                        ..first_overload
                    },
                    true,
                )
            }
        }
    }

    fn disambiguate_overload_results(&self, results: &[Type]) -> Option<usize> {
        // Step 6: does there exist a return type that is consistent with all materializations of
        // every other return type? If so, use this return type. Else, return Any.
        //
        // We check materializations so that we end up with the most "general" return type. E.g.,
        // if the candidates are `A[None]` and `A[Any]`, we want to select `A[Any]`.
        //
        // First, find a candidate return type.
        let mut candidate = 0;
        for (i, result) in results.iter().enumerate().skip(1) {
            if !self.is_consistent(&result.materialize(), &results[candidate]) {
                candidate = i;
            }
        }
        // We've already checked every return type after the candidate.
        // Check every return type before the candidate.
        for result in results.iter().take(candidate) {
            if !self.is_consistent(&result.materialize(), &results[candidate]) {
                return None;
            }
        }
        Some(candidate)
    }

    /// Collect placeholder vars from self_obj and Type-valued arguments.
    fn collect_placeholder_vars(
        &self,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
    ) -> Vec<Var> {
        let mut placeholder_vars: Vec<Var> = Vec::new();
        let mut collect = |ty: &Type| {
            for var in ty.collect_maybe_placeholder_vars() {
                if !placeholder_vars.contains(&var) {
                    placeholder_vars.push(var);
                }
            }
        };
        if let Some(obj) = self_obj {
            collect(obj);
        }
        for arg in args {
            if let CallArg::Arg(TypeOrExpr::Type(ty, _))
            | CallArg::Star(TypeOrExpr::Type(ty, _), _) = arg
            {
                collect(ty);
            }
        }
        for kw in keywords {
            if let TypeOrExpr::Type(ty, _) = &kw.value {
                collect(ty);
            }
        }
        placeholder_vars
    }

    fn call_overload<'c>(
        &self,
        callable: &'c TargetWithTParams<Function>,
        metadata: &FuncMetadata,
        shape_transform: Option<&ShapeTransform>,
        self_obj: Option<&Type>,
        args: &[CallArg],
        keywords: &[CallKeyword],
        arguments_range: TextRange,
        hint: Option<HintRef>,
        contextually_opaque_defaults: Option<&SmallSet<Quantified>>,
        ctor_targs: &Option<&mut TArgs>,
    ) -> CalledOverload<'c> {
        // Create a copy of the class type arguments (if any) that should be filled in by this call.
        // The `callable_infer` call below will fill in this copy with the type arguments set
        // by the current overload, and we'll later use the copy to fill in the original
        // ctor_targs if this overload is chosen.
        let mut overload_ctor_targs = ctor_targs.as_ref().map(|x| (**x).clone());
        let tparams = callable.0.as_deref();

        // `@uses_shape_dsl` may sit on a single overload (e.g. a shape-DSL variant that
        // follows a plain-TypeVar fast path). The set-wide `shape_transform` only carries
        // the first overload's decorator, so prefer this overload's own transform and only
        // fall back to the set-wide one (e.g. an implementation-level decorator).
        let shape_transform = callable
            .1
            .metadata
            .flags
            .shape_transform
            .as_deref()
            .or(shape_transform);

        let arg_errors = self.error_collector();
        let call_errors = self.error_collector();
        let (res, specialization_errors, return_type_errors, argmap, defaults_used, table) = self
            .callable_infer(
                callable.1.signature.clone(),
                Some(&metadata.kind),
                shape_transform,
                tparams,
                self_obj.cloned(),
                args,
                keywords,
                arguments_range,
                &arg_errors,
                &call_errors,
                // We intentionally drop the context here, as arg errors don't need it,
                // and if there are any call errors, we'll log a "No matching overloads"
                // error with the necessary context.
                None,
                hint,
                contextually_opaque_defaults,
                overload_ctor_targs.as_mut(),
            );
        CalledOverload {
            func: callable,
            res,
            ctor_targs: overload_ctor_targs,
            table,
            arg_errors,
            call_errors,
            specialization_errors,
            return_type_errors,
            defaults_used,
            argmap,
        }
    }
}

//! How a Bevy-like ECS learns what a system touches, from its signature alone.
//!
//! Run it and read the output:
//!
//! ```text
//! cargo test --test ecs_systems -- --nocapture
//! ```
//!
//! # The question this file answers
//!
//! Given an ordinary Rust function:
//!
//! ```text
//! fn movement(query: Query<(&mut Position, &Velocity), Without<Dead>>) { }
//! ```
//!
//! and nothing but `register_system("movement", movement)`, how does the engine
//! work out that this system **writes** `Position`, **reads** `Velocity`, and
//! wants only entities **without** `Dead`?
//!
//! The short answer: it does not inspect the function at all. It inspects the
//! function's *type*. Every piece of that signature is a type, every one of those
//! types implements a trait, and each trait implementation knows how to describe
//! itself. The compiler picks the implementations; the descriptions are collected
//! into an ordinary struct; and that struct is what the scheduler reasons about.
//!
//! **This is not reflection.** Nothing looks up type information at run time.
//! Every choice of implementation is made while compiling, and what survives into
//! the running program is a `HashSet` of `TypeId`s that the compiler arranged to
//! have filled in.
//!
//! **Nor is it function overloading.** Rust has no overloading. There is one
//! `register_system` function; it is generic, and the compiler *solves* for which
//! generic implementation fits the argument's type. That distinction is the heart
//! of the trick and is explained at [`IntoSystem`].
//!
//! # What is deliberately left out
//!
//! Almost everything an ECS actually does. There are no archetypes worth the
//! name, no storage abstraction, no change detection, no commands, no real
//! parallel executor. Component storage is four concrete hash maps, because the
//! point here is the *metadata*, and a generic storage layer would bury it.

use std::any::{TypeId, type_name};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

// ===========================================================================
// 1. Components
// ===========================================================================

/// A marker trait for types that can be attached to an entity.
///
/// # What `'static` means here, and why it is needed
///
/// `'static` as a *bound on a type* (as opposed to a reference lifetime) means
/// "this type contains no borrowed data". `Position` qualifies; `&'a str` does
/// not, because it borrows for some shorter `'a`.
///
/// It is required because [`TypeId::of`] demands it. A `TypeId` is a unique
/// identifier for a type, and a type that borrows would need a different
/// identity per lifetime — so the standard library only hands them out for
/// `'static` types. Since identifying component types is the whole mechanism,
/// `'static` is not optional.
///
/// # On the blanket implementation
///
/// It is tempting to write:
///
/// ```text
/// impl<T: 'static> Component for T {}
/// ```
///
/// which would make *every* owned type a component with no further work. That is
/// deliberately not done here, for two reasons:
///
/// 1. **It gives up all checking.** `Query<&u32>` and even `Query<&Query<...>>`
///    would compile, meaning nothing. Implementing the trait by hand is the only
///    thing that says "this type is intended as a component".
/// 2. **It cannot be narrowed later.** Once every type satisfies the trait,
///    adding a required method or an associated constant breaks every type in the
///    program at once. A trait implemented for four types can gain a method for
///    the cost of four lines.
///
/// Bevy takes the same view: its `Component` is derived, not blanket.
trait Component: 'static {}

/// Where an entity is.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}

/// How fast it is going.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Velocity {
    x: f32,
    y: f32,
}

/// How much punishment it has left.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Health(u32);

/// A component with no data at all, used purely as a label.
///
/// Attaching it says something about the entity without storing anything, which
/// is what makes it useful in a [`Without`] filter.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Dead;

impl Component for Position {}
impl Component for Velocity {}
impl Component for Health {}
impl Component for Dead {}

/// A component type's identity, plus a name so that the output can be read.
///
/// # Why both
///
/// [`TypeId`] is what the program compares — it is a number, and comparing two of
/// them is fast and exact. [`type_name`] is a string the compiler produces for
/// diagnostics; it is *not* guaranteed unique or stable between compiler
/// versions, so it must never be used to decide anything. It is here so a human
/// can read "Position" instead of a hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ComponentId {
    /// What the program compares.
    type_id: TypeId,
    /// What a person reads.
    name: &'static str,
}

impl ComponentId {
    /// The identity of one component type.
    ///
    /// `ComponentId::of::<Position>()` is resolved entirely at compile time: the
    /// compiler knows which type `T` is, so it knows which `TypeId` to embed.
    fn of<T: Component>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            // `type_name` gives a full path such as `ecs_systems::Position`; the
            // last segment is what is worth printing.
            name: type_name::<T>()
                .rsplit("::")
                .next()
                .unwrap_or_else(|| type_name::<T>()),
        }
    }
}

impl std::fmt::Debug for ComponentId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name)
    }
}

// ===========================================================================
// 2. Entities and a very small world
// ===========================================================================

/// An entity is only a number. It owns nothing; components are stored elsewhere
/// and found by this key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Entity(u32);

/// Component storage, kept as deliberately dull as possible.
///
/// One concrete map per component type. A real ECS stores components generically
/// — Bevy groups entities with identical component sets into *archetypes* and
/// stores each component in a contiguous column — but any of that here would
/// obscure the part worth learning. Four maps are enough to run one system at the
/// bottom of the file and show that the metadata told the truth.
#[derive(Default)]
struct World {
    positions: HashMap<Entity, Position>,
    velocities: HashMap<Entity, Velocity>,
    healths: HashMap<Entity, Health>,
    dead: HashSet<Entity>,
    next_entity: u32,
}

impl World {
    /// A fresh entity with nothing attached.
    fn spawn(&mut self) -> Entity {
        let entity = Entity(self.next_entity);
        self.next_entity += 1;

        entity
    }

    /// Which component types an entity actually has.
    ///
    /// This is the stand-in for an archetype: in a real engine the *group* would
    /// know its components and every entity in it would share them, which is what
    /// makes filtering cheap. Here it is worked out per entity.
    fn archetype_of(&self, entity: Entity) -> ArchetypeInfo {
        let mut components: HashSet<TypeId> = HashSet::new();

        if self.positions.contains_key(&entity) {
            components.insert(TypeId::of::<Position>());
        }
        if self.velocities.contains_key(&entity) {
            components.insert(TypeId::of::<Velocity>());
        }
        if self.healths.contains_key(&entity) {
            components.insert(TypeId::of::<Health>());
        }
        if self.dead.contains(&entity) {
            components.insert(TypeId::of::<Dead>());
        }

        ArchetypeInfo { components }
    }

    /// Every entity that has been spawned.
    fn entities(&self) -> Vec<Entity> {
        (0..self.next_entity).map(Entity).collect()
    }
}

/// What component types a group of entities has.
///
/// A filter is asked about this and answers yes or no. Only `TypeId`s are needed,
/// since a filter never looks at a component's *value* — only whether it is
/// present.
struct ArchetypeInfo {
    components: HashSet<TypeId>,
}

impl ArchetypeInfo {
    /// Whether this group has a particular component type.
    fn has<T: Component>(&self) -> bool {
        self.components.contains(&TypeId::of::<T>())
    }
}

// ===========================================================================
// 3. The metadata that all of this exists to produce
// ===========================================================================

/// What one system touches, and what it requires of the entities it sees.
///
/// This is the *only* thing that crosses from compile time into run time. Every
/// trait below exists to fill one of these in. Once it is filled in, the
/// scheduler needs nothing else — it never looks at the system function again.
///
/// # Why access and filtering are kept apart
///
/// `reads` and `writes` decide whether two systems may run at the same time:
/// two systems that both write `Position` must not. `required` and `excluded`
/// decide which *entities* a system sees, and say nothing about conflicts — two
/// systems reading `Position` with opposite filters still do not conflict,
/// because reading never conflicts with reading.
///
/// Keeping them in separate sets makes that distinction explicit rather than
/// something the conflict rule has to remember.
#[derive(Default)]
struct SystemAccess {
    /// Component types read, from `&T`.
    reads: HashSet<ComponentId>,
    /// Component types written, from `&mut T`.
    writes: HashSet<ComponentId>,
    /// Component types an entity must have, from `With<T>`.
    required: HashSet<ComponentId>,
    /// Component types an entity must not have, from `Without<T>`.
    excluded: HashSet<ComponentId>,
}

impl SystemAccess {
    /// Records a read of `T`.
    fn add_read<T: Component>(&mut self) {
        self.reads.insert(ComponentId::of::<T>());
    }

    /// Records a write of `T`.
    ///
    /// A write implies the ability to read, but the two sets are kept disjoint
    /// here so that the conflict rule below reads plainly. A real engine usually
    /// does the same and treats a write as strictly stronger.
    fn add_write<T: Component>(&mut self) {
        self.writes.insert(ComponentId::of::<T>());
    }

    /// Records that entities must have `T`.
    fn add_required<T: Component>(&mut self) {
        self.required.insert(ComponentId::of::<T>());
    }

    /// Records that entities must not have `T`.
    fn add_excluded<T: Component>(&mut self) {
        self.excluded.insert(ComponentId::of::<T>());
    }
}

/// Sorted names, so the printed output is the same on every run.
///
/// A `HashSet` iterates in an unspecified order; sorting by name is the cheapest
/// way to make the demonstration readable and repeatable.
fn sorted_names(set: &HashSet<ComponentId>) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = set.iter().map(|component| component.name).collect();
    names.sort_unstable();

    names
}

// ===========================================================================
// 4. QueryData: how `&T` and `&mut T` describe themselves
// ===========================================================================

/// A type that can say which components it reads and writes.
///
/// # Why `&T` and `&mut T` can have separate implementations
///
/// Because they are *different types*. `&Position` and `&mut Position` are as
/// distinct to the compiler as `u8` and `u16`, so a trait may be implemented once
/// for each with entirely different bodies. That is the whole mechanism: the
/// mutability the programmer wrote in the signature is part of the type, and the
/// type selects the implementation that records a write rather than a read.
///
/// Nothing is being detected. `&mut Position` does not "look mutable" to any
/// run-time check; it simply *is* a type whose implementation calls
/// [`SystemAccess::add_write`].
///
/// # Why the method takes no `self`
///
/// There is no value to inspect. The description depends only on the type, so it
/// is an associated function called as `<&mut Position as QueryData>::describe`.
/// The compiler resolves that at the call site, and what remains in the compiled
/// program is a direct call — no dispatch, no lookup.
trait QueryData {
    /// Adds this piece of the query's access to `access`.
    fn describe(access: &mut SystemAccess);
}

/// A shared reference records a read.
///
/// `impl<T: Component> QueryData for &T` reads as: for every type `T` that is a
/// `Component`, the type `&T` implements `QueryData` this way. So one written
/// implementation covers `&Position`, `&Velocity`, and every component yet to be
/// invented — that is what a *generic implementation* buys.
impl<T: Component> QueryData for &T {
    fn describe(access: &mut SystemAccess) {
        access.add_read::<T>();
    }
}

/// A mutable reference records a write.
impl<T: Component> QueryData for &mut T {
    fn describe(access: &mut SystemAccess) {
        access.add_write::<T>();
    }
}

/// Generates `QueryData` for tuples of a given length.
///
/// # Why a macro is needed at all
///
/// Rust has no *variadic generics*: there is no way to write one implementation
/// covering tuples of every length, because `(A,)`, `(A, B)` and `(A, B, C)` are
/// unrelated types with no shared "length" the language can abstract over. Each
/// arity therefore needs its own implementation, and they are all identical
/// except for how many type parameters they mention.
///
/// A macro is the right tool precisely because the repetition is mechanical. This
/// is not cleverness — it is the same four lines written four times, with the
/// compiler doing the typing.
///
/// # Reading the macro
///
/// - `$($name:ident),*` captures a comma-separated list of identifiers, so one
///   invocation can pass `A, B, C`.
/// - `$($name: QueryData),*` expands that list into bounds:
///   `A: QueryData, B: QueryData, C: QueryData`.
/// - `$($name::describe(access);)*` expands it into one statement per element.
///
/// Bevy does exactly this, for tuples up to sixteen or so; the limit is arbitrary
/// and chosen to keep compile times sane.
macro_rules! implement_query_data_for_tuple {
    ($($name:ident),*) => {
        impl<$($name: QueryData),*> QueryData for ($($name,)*) {
            fn describe(access: &mut SystemAccess) {
                // One call per element of the tuple. For `(&mut Position,
                // &Velocity)` this expands to two calls: one recording a write,
                // one recording a read.
                $($name::describe(access);)*
            }
        }
    };
}

// Lengths one through four. The trailing comma in `($($name,)*)` above is what
// makes the one-element case `(A,)` rather than the plain `A` that parentheses
// would otherwise mean.
implement_query_data_for_tuple!(A);
implement_query_data_for_tuple!(A, B);
implement_query_data_for_tuple!(A, B, C);
implement_query_data_for_tuple!(A, B, C, D);

// ===========================================================================
// 5. QueryFilter: With, Without, and tuples meaning AND
// ===========================================================================

/// A condition on which entities a query sees.
///
/// Two jobs, and it is worth seeing why they are different:
///
/// - [`QueryFilter::matches`] runs at *run time*, against a real group of
///   entities, and decides whether to visit them.
/// - [`QueryFilter::describe`] runs at *registration time* and records the filter
///   into the metadata, so that a human — or a tool — can see what a system
///   demands without running it.
trait QueryFilter {
    /// Whether a group of entities passes this filter.
    fn matches(archetype: &ArchetypeInfo) -> bool;

    /// Records this filter's demands into the metadata.
    fn describe(access: &mut SystemAccess);
}

/// Requires that entities have `T`.
///
/// # Why [`PhantomData`]
///
/// `With<Health>` needs to mention `Health` in its type — that is the entire
/// point of it — but it stores no `Health` value. Rust rejects a type parameter
/// that is never used in a field, because it could not work out the type's
/// variance or which auto traits it should have. `PhantomData<T>` is a
/// zero-sized field that says "behave as though a `T` were stored here", which
/// satisfies the compiler and costs no memory at all.
///
/// `With<Health>` is therefore a type carrying information and no data — exactly
/// what is wanted, since the information is all that is ever read.
struct With<T>(PhantomData<T>);

/// Requires that entities do **not** have `T`.
struct Without<T>(PhantomData<T>);

impl<T: Component> QueryFilter for With<T> {
    fn matches(archetype: &ArchetypeInfo) -> bool {
        archetype.has::<T>()
    }

    fn describe(access: &mut SystemAccess) {
        access.add_required::<T>();
    }
}

impl<T: Component> QueryFilter for Without<T> {
    fn matches(archetype: &ArchetypeInfo) -> bool {
        !archetype.has::<T>()
    }

    fn describe(access: &mut SystemAccess) {
        access.add_excluded::<T>();
    }
}

/// The empty filter: no condition, so every entity passes.
///
/// This is what makes `Query<(&Position,)>` work with no second type argument —
/// the default filter is `()`, and here is its implementation.
impl QueryFilter for () {
    fn matches(_archetype: &ArchetypeInfo) -> bool {
        true
    }

    fn describe(_access: &mut SystemAccess) {}
}

/// Passes when **either** filter passes.
///
/// Included to show a case the metadata cannot fully express — see the
/// implementation.
struct Or<T>(PhantomData<T>);

impl<A: QueryFilter, B: QueryFilter> QueryFilter for Or<(A, B)> {
    fn matches(archetype: &ArchetypeInfo) -> bool {
        A::matches(archetype) || B::matches(archetype)
    }

    /// Deliberately records nothing.
    ///
    /// `required` and `excluded` are *sets*, and a set can only express "all of
    /// these". `Or` is a disjunction — "either of these" — and there is no way to
    /// write that as a set of required components. Recording either side would be
    /// a lie: it would claim a requirement that entities need not meet.
    ///
    /// So the filtering still works (`matches` is correct), while the metadata
    /// simply says less than the filter does. Bevy meets the same wall, and this
    /// is why filter metadata is generally used for documentation and debugging
    /// rather than for scheduling decisions.
    fn describe(_access: &mut SystemAccess) {}
}

/// Generates `QueryFilter` for tuples, where a tuple means logical AND.
///
/// `(With<A>, With<B>, Without<C>)` therefore means `A and B and not C`: every
/// element must match, and every element's demands are recorded.
///
/// Same reason for the macro as before — no variadic generics — and the body is
/// slightly more interesting than the `QueryData` one, because the `matches`
/// arm has to combine results with `&&` rather than just running statements.
macro_rules! implement_query_filter_for_tuple {
    ($($name:ident),*) => {
        impl<$($name: QueryFilter),*> QueryFilter for ($($name,)*) {
            fn matches(archetype: &ArchetypeInfo) -> bool {
                // Expands to `true && A::matches(..) && B::matches(..)`. The
                // leading `true` is what makes the expansion valid for any
                // length, including one.
                true $(&& $name::matches(archetype))*
            }

            fn describe(access: &mut SystemAccess) {
                $($name::describe(access);)*
            }
        }
    };
}

implement_query_filter_for_tuple!(A);
implement_query_filter_for_tuple!(A, B);
implement_query_filter_for_tuple!(A, B, C);
implement_query_filter_for_tuple!(A, B, C, D);

// ===========================================================================
// 6. Query, and the SystemParam trait
// ===========================================================================

/// A request for entities and the components to touch on them.
///
/// `Q` says which components, `F` which entities. `F` defaults to `()`, so
/// `Query<&Position>` means `Query<&Position, ()>`.
///
/// # It holds nothing
///
/// In this file a `Query` is purely a type-level description: two `PhantomData`
/// fields and no data. That is enough for everything above — the metadata comes
/// from `Q` and `F`, not from any value.
///
/// A real `Query` also borrows the world and hands out component references, and
/// the lifetimes that requires are the single largest source of complexity in an
/// ECS. Leaving it out is the main simplification in this file; the execution
/// demonstration at the bottom does its work directly on the [`World`] instead.
struct Query<Q, F = ()> {
    data: PhantomData<Q>,
    filter: PhantomData<F>,
}

impl<Q, F> Query<Q, F> {
    /// A query value, needed only because the execution demonstration wants
    /// something to pass.
    fn new() -> Self {
        Self {
            data: PhantomData,
            filter: PhantomData,
        }
    }
}

/// Anything that may appear as a parameter of a system function.
///
/// This is the seam that makes the whole design extensible: to add a new kind of
/// system parameter, implement this one trait for it. Nothing else changes — not
/// `register_system`, not the scheduler, not the conflict rule.
trait SystemParam {
    /// Adds this parameter's access to the system's metadata.
    fn describe(access: &mut SystemAccess);
}

/// A query describes itself by asking its two type arguments to describe
/// themselves.
///
/// Notice that this implementation contains no knowledge of `Position`,
/// `&mut T`, `With`, or anything else concrete. It delegates, and the compiler
/// has already worked out what to delegate *to* from the types `Q` and `F`.
impl<Q: QueryData, F: QueryFilter> SystemParam for Query<Q, F> {
    fn describe(access: &mut SystemAccess) {
        Q::describe(access);
        F::describe(access);
    }
}

/// Shared access to a resource: one global value, not attached to an entity.
///
/// Never constructed anywhere in this file — only named in a signature, which is
/// all the mechanism needs.
#[allow(dead_code)]
///
/// Included as a second kind of `SystemParam`, to show that the mechanism is not
/// about queries specifically. A resource's access could be tracked in its own
/// pair of sets; here it borrows the component sets, which is a simplification
/// worth noticing — a real engine keeps resource access separate so that a
/// resource and a component of the same type cannot be confused.
struct Res<T>(PhantomData<T>);

/// Exclusive access to a resource.
#[allow(dead_code)]
struct ResMut<T>(PhantomData<T>);

impl<T: Component> SystemParam for Res<T> {
    fn describe(access: &mut SystemAccess) {
        access.add_read::<T>();
    }
}

impl<T: Component> SystemParam for ResMut<T> {
    fn describe(access: &mut SystemAccess) {
        access.add_write::<T>();
    }
}

// ===========================================================================
// 7. IntoSystem: the part that turns a plain function into a system
// ===========================================================================

/// A thing that can be registered as a system.
///
/// # The `Params` type parameter is the trick
///
/// The obvious design would be `trait IntoSystem { … }` with implementations for
/// `F: FnMut(A)`, `F: FnMut(A, B)`, and so on. **That does not compile.** Rust
/// forbids two implementations of one trait for what might be the same type, and
/// it cannot rule out a type implementing both `FnMut(A)` and `FnMut(A, B)`. The
/// two implementations would overlap, and overlap is rejected.
///
/// Adding a type parameter fixes it. `IntoSystem<(A,)>` and `IntoSystem<(A, B)>`
/// are *different traits* as far as coherence is concerned, so a single function
/// type may implement both without conflict. The parameter is never used in the
/// body; it exists only to keep the implementations apart.
///
/// This is precisely what Bevy does, and why its error messages sometimes mention
/// a mysterious `Params` you never wrote.
///
/// # How the compiler chooses
///
/// At `register_system("movement", movement)`:
///
/// 1. The compiler knows `movement`'s type: a function item whose signature is
///    `fn(Query<(&mut Position, &Velocity), Without<Dead>>)`.
/// 2. `register_system` demands `F: IntoSystem<Params>` for *some* `Params`.
/// 3. Only one implementation can fit: the one-parameter one, with
///    `Params = (Query<(&mut Position, &Velocity), Without<Dead>>,)`.
/// 4. Fitting it requires that parameter to implement [`SystemParam`], which
///    sends the compiler to the `Query` implementation, which requires
///    `(&mut Position, &Velocity)` to implement [`QueryData`] and
///    `Without<Dead>` to implement [`QueryFilter`] — and so on down to `&mut
///    Position`.
///
/// Every step is a search for a matching implementation, carried out while
/// compiling. This is *type inference and trait resolution*, not overloading:
/// there is one `register_system`, and the compiler is solving for its generic
/// arguments rather than choosing between several functions of the same name.
///
/// If any step fails — a parameter that is not a `SystemParam`, a query element
/// that is neither `&T` nor `&mut T` — the result is a compile error naming the
/// unsatisfied bound. The system is rejected before the program exists.
trait IntoSystem<Params> {
    /// The access this system's signature implies.
    ///
    /// No `self`: the answer depends only on the function's *type*, so no value
    /// is needed. That is why [`register_system`] can ignore the function it is
    /// handed.
    fn access() -> SystemAccess;
}

/// Generates `IntoSystem` for functions of a given arity.
///
/// The same absence of variadic generics as before, now applied to function
/// parameters rather than tuple elements: `FnMut(A)` and `FnMut(A, B)` are
/// unrelated bounds, so each arity needs its own implementation.
///
/// Note what the generated implementation does *not* do: it never calls the
/// function. The `FnMut` bound is there only to let the compiler read the
/// parameter types out of the function's signature.
macro_rules! implement_into_system_for_arity {
    ($($param:ident),*) => {
        impl<Function, $($param),*> IntoSystem<($($param,)*)> for Function
        where
            // What makes this apply to plain `fn` items and closures alike: both
            // implement `FnMut`.
            Function: FnMut($($param),*),
            // Every parameter must be able to describe itself.
            $($param: SystemParam),*
        {
            fn access() -> SystemAccess {
                // `mut` is allowed but unused in the zero-parameter expansion,
                // where there is nothing to add; the attribute keeps that one
                // case quiet without weakening the others.
                #[allow(unused_mut)]
                let mut access = SystemAccess::default();

                // One call per parameter. This is where a signature becomes
                // data.
                $($param::describe(&mut access);)*

                access
            }
        }
    };
}

// Arities zero through four. Zero is worth having: a system that takes nothing
// accesses nothing, and so conflicts with nothing.
implement_into_system_for_arity!();
implement_into_system_for_arity!(A);
implement_into_system_for_arity!(A, B);
implement_into_system_for_arity!(A, B, C);
implement_into_system_for_arity!(A, B, C, D);

/// A system as the scheduler sees it: a name and what it touches.
///
/// The function itself is gone. Everything needed to schedule it has been
/// extracted, and for this demonstration nothing more is kept — a real engine
/// would of course also store something callable.
struct RegisteredSystem {
    name: &'static str,
    access: SystemAccess,
}

/// Registers a system, extracting its access from its signature.
///
/// # Why the function argument is ignored
///
/// `_system` is never used. It is a parameter purely so that the compiler has a
/// value whose *type* it can inspect to infer `Function` and `Params`. Writing
/// `register_system::<_, _>("movement")` with no function would leave the
/// compiler nothing to solve from.
///
/// So the value is discarded and the type is kept. That inversion — the argument
/// exists for its type, not its value — is the clearest single illustration of
/// what is going on in this file.
fn register_system<Function, Params>(
    name: &'static str,
    _system: Function,
) -> RegisteredSystem
where
    Function: IntoSystem<Params>,
{
    RegisteredSystem {
        name,
        access: Function::access(),
    }
}

// ===========================================================================
// 8. Conflict detection
// ===========================================================================

/// Whether two systems may not run at the same time.
///
/// # The rule
///
/// Two systems conflict when one writes something the other touches:
///
/// - one writes `T` and the other reads `T`
/// - one writes `T` and the other also writes `T`
///
/// and nothing else. In particular **read and read never conflict**: any number
/// of systems may read `Position` at once, because none of them can observe a
/// change that is not there.
///
/// This is exactly Rust's own borrowing rule — many shared borrows, or one
/// exclusive one — lifted from variables to whole systems. That is not a
/// coincidence: it is the same underlying fact about when concurrent access is
/// safe.
///
/// # What the filters do not do
///
/// `required` and `excluded` are not consulted. It is tempting to think that two
/// systems writing `Position`, one `With<Dead>` and one `Without<Dead>`, cannot
/// collide — they see disjoint entities. That is true, and this rule still calls
/// them a conflict.
///
/// The reason is that being right here is harder than it looks: filters can
/// overlap in ways that are undecidable in general, and `Or` (above) cannot even
/// be recorded. Declaring a conflict is always *safe*; missing one is a data
/// race. So the rule is deliberately conservative, and says so.
fn conflicts(first: &SystemAccess, second: &SystemAccess) -> bool {
    let writes_what_other_reads = first.writes.intersection(&second.reads).next().is_some();
    let reads_what_other_writes = first.reads.intersection(&second.writes).next().is_some();
    let both_write = first.writes.intersection(&second.writes).next().is_some();

    writes_what_other_reads || reads_what_other_writes || both_write
}

// ===========================================================================
// 9. Grouping, as a scheduler might
// ===========================================================================

/// Places systems into groups where nothing inside a group conflicts.
///
/// Greedy and simple: walk the systems in order, drop each into the first group
/// that will take it, and start a new group when none will. Systems in one group
/// could in principle run at the same time.
///
/// # How far this is from a real scheduler
///
/// A long way. A real engine — Bevy among them — additionally handles:
///
/// - **explicit ordering**, where one system must precede another regardless of
///   data access,
/// - **exclusive systems** that need the whole world,
/// - **run conditions** deciding whether a system runs at all this tick,
/// - **work stealing**, so that a thread finishing early picks up other work
///   rather than waiting for its group,
/// - and the observation that groups are the wrong model in the first place: what
///   matters is a dependency *graph*, not a sequence of barriers.
///
/// The purpose here is only to show that access metadata derived from a
/// signature is enough to start reasoning about any of it.
fn group_systems(systems: &[RegisteredSystem]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for (index, system) in systems.iter().enumerate() {
        // The first group with nothing this system conflicts with.
        let fits = groups.iter_mut().find(|group| {
            group
                .iter()
                .all(|other| !conflicts(&system.access, &systems[*other].access))
        });

        match fits {
            Some(group) => group.push(index),
            None => groups.push(vec![index]),
        }
    }

    groups
}

// ===========================================================================
// 10. Example systems
// ===========================================================================
//
// These are ordinary functions. Nothing is implemented for them by hand; they
// qualify as systems purely because their parameter types satisfy the bounds.
//
// The bodies are empty because the metadata is what is being demonstrated. The
// one system that actually does work is at the very bottom.

/// Writes `Position`, reads `Velocity`, and skips dead entities.
fn movement(_query: Query<(&mut Position, &Velocity), Without<Dead>>) {}

/// Reads `Position` only.
fn render_positions(_query: Query<&Position>) {}

/// Writes `Health`.
fn damage(_query: Query<&mut Health>) {}

/// Reads `Health`.
fn inspect_health(_query: Query<&Health>) {}

/// Two parameters, to exercise a different arity — and a filter tuple meaning
/// AND: entities with `Health` but not `Dead`.
fn regenerate(
    _living: Query<&mut Health, (With<Health>, Without<Dead>)>,
    _positions: Query<&Position>,
) {
}

/// Takes nothing, so it conflicts with nothing.
fn tick_clock() {}

/// A resource parameter rather than a query, showing the seam is general.
fn count_entities(_count: ResMut<Health>) {}

// ===========================================================================
// 11. The demonstration
// ===========================================================================

#[test]
fn systems_describe_themselves_from_their_signatures() {
    // Registration. Note that no type annotations are needed anywhere: the
    // compiler infers `Params` for each function from its signature, and follows
    // the trait bounds down to `&T` and `&mut T`.
    let systems: Vec<RegisteredSystem> = vec![
        register_system("movement", movement),
        register_system("render_positions", render_positions),
        register_system("damage", damage),
        register_system("inspect_health", inspect_health),
        register_system("regenerate", regenerate),
        register_system("tick_clock", tick_clock),
        register_system("count_entities", count_entities),
    ];

    println!("\n=== What each signature revealed ===\n");

    for system in &systems {
        println!("{}", system.name);
        println!("    reads    {:?}", sorted_names(&system.access.reads));
        println!("    writes   {:?}", sorted_names(&system.access.writes));
        println!("    requires {:?}", sorted_names(&system.access.required));
        println!("    excludes {:?}", sorted_names(&system.access.excluded));
    }

    // --- The metadata is correct, asserted rather than merely printed. -------

    let by_name = |wanted: &str| -> &SystemAccess {
        &systems
            .iter()
            .find(|system| system.name == wanted)
            .expect("registered above")
            .access
    };

    let movement_access = by_name("movement");
    assert!(
        movement_access.writes.contains(&ComponentId::of::<Position>()),
        "`&mut Position` should have recorded a write"
    );
    assert!(
        movement_access.reads.contains(&ComponentId::of::<Velocity>()),
        "`&Velocity` should have recorded a read"
    );
    assert!(
        movement_access.excluded.contains(&ComponentId::of::<Dead>()),
        "`Without<Dead>` should have recorded an exclusion"
    );
    assert!(
        movement_access.required.is_empty(),
        "there was no `With` in the signature"
    );

    // A filter tuple means AND, and both halves were recorded.
    let regenerate_access = by_name("regenerate");
    assert!(regenerate_access.required.contains(&ComponentId::of::<Health>()));
    assert!(regenerate_access.excluded.contains(&ComponentId::of::<Dead>()));
    assert!(regenerate_access.writes.contains(&ComponentId::of::<Health>()));
    assert!(
        regenerate_access.reads.contains(&ComponentId::of::<Position>()),
        "the second parameter contributed too"
    );

    // A system with no parameters accesses nothing.
    let clock_access = by_name("tick_clock");
    assert!(clock_access.reads.is_empty() && clock_access.writes.is_empty());

    println!("\n=== Conflicts ===\n");

    for (index, first) in systems.iter().enumerate() {
        for second in systems.iter().skip(index + 1) {
            let verdict = if conflicts(&first.access, &second.access) {
                "CONFLICT"
            } else {
                "no conflict"
            };

            println!("{:>16} vs {:<16} {verdict}", first.name, second.name);
        }
    }

    // One writes Position, the other reads it.
    assert!(conflicts(by_name("movement"), by_name("render_positions")));

    // Unrelated component types.
    assert!(!conflicts(by_name("movement"), by_name("damage")));

    // Both only read, so any number of them may run together.
    assert!(!conflicts(by_name("render_positions"), by_name("inspect_health")));

    // Write against write, on the same type.
    assert!(conflicts(by_name("damage"), by_name("regenerate")));

    // A system touching nothing conflicts with nothing.
    assert!(!conflicts(by_name("tick_clock"), by_name("movement")));

    // A resource parameter participates in the same rule as a query.
    assert!(conflicts(by_name("count_entities"), by_name("inspect_health")));

    println!("\n=== Groups that could run in parallel ===\n");

    let groups = group_systems(&systems);

    for (number, group) in groups.iter().enumerate() {
        println!("Group {number}:");

        for index in group {
            println!("    {}", systems[*index].name);
        }
    }

    // Whatever the grouping turns out to be, the property that matters must hold:
    // nothing inside a group conflicts with anything else inside it.
    for group in &groups {
        for (position, first) in group.iter().enumerate() {
            for second in group.iter().skip(position + 1) {
                assert!(
                    !conflicts(&systems[*first].access, &systems[*second].access),
                    "{} and {} were grouped together but conflict",
                    systems[*first].name,
                    systems[*second].name
                );
            }
        }
    }

    // And every system was placed exactly once.
    let placed: usize = groups.iter().map(Vec::len).sum();
    assert_eq!(placed, systems.len());
}

#[test]
fn filters_decide_which_entities_a_system_would_see() {
    // The other half of a filter's job: `matches` at run time, against a real
    // group of entities.
    let mut world = World::default();

    let walker = world.spawn();
    world.positions.insert(walker, Position { x: 0.0, y: 0.0 });
    world.velocities.insert(walker, Velocity { x: 1.0, y: 0.0 });
    world.healths.insert(walker, Health(10));

    let corpse = world.spawn();
    world.positions.insert(corpse, Position { x: 5.0, y: 5.0 });
    world.velocities.insert(corpse, Velocity { x: 9.0, y: 9.0 });
    world.dead.insert(corpse);

    let living = world.archetype_of(walker);
    let dead = world.archetype_of(corpse);

    // `Without<Dead>` passes the walker and rejects the corpse.
    assert!(<Without<Dead> as QueryFilter>::matches(&living));
    assert!(!<Without<Dead> as QueryFilter>::matches(&dead));

    // `With<Health>` the other way about.
    assert!(<With<Health> as QueryFilter>::matches(&living));
    assert!(!<With<Health> as QueryFilter>::matches(&dead));

    // A tuple means AND: alive *and* has health.
    type Living = (With<Health>, Without<Dead>);
    assert!(<Living as QueryFilter>::matches(&living));
    assert!(!<Living as QueryFilter>::matches(&dead));

    // The empty filter passes everything.
    assert!(<() as QueryFilter>::matches(&dead));

    // `Or` passes when either side does — and note it recorded nothing in the
    // metadata, for the reason its implementation explains.
    type EitherWay = Or<(With<Dead>, With<Health>)>;
    assert!(<EitherWay as QueryFilter>::matches(&living));
    assert!(<EitherWay as QueryFilter>::matches(&dead));

    let mut access = SystemAccess::default();
    <EitherWay as QueryFilter>::describe(&mut access);
    assert!(
        access.required.is_empty(),
        "a disjunction cannot be written as a set of requirements"
    );
}

#[test]
fn one_system_actually_runs() {
    // The metadata said `movement` writes Position, reads Velocity, and skips
    // entities with Dead. This does exactly that, by hand, to show the
    // description was true.
    //
    // It works on the `World` directly rather than through `Query`, because a
    // real `Query` would have to borrow the world and hand out references — and
    // the lifetimes involved are the one thing that would make this file hard to
    // read. That omission is the price of clarity here.
    let mut world = World::default();

    let walker = world.spawn();
    world.positions.insert(walker, Position { x: 0.0, y: 0.0 });
    world.velocities.insert(walker, Velocity { x: 2.0, y: 3.0 });

    let corpse = world.spawn();
    world.positions.insert(corpse, Position { x: 100.0, y: 100.0 });
    world.velocities.insert(corpse, Velocity { x: 5.0, y: 5.0 });
    world.dead.insert(corpse);

    // The same filter type the signature used, applied for real.
    type Filter = Without<Dead>;

    for entity in world.entities() {
        if !<Filter as QueryFilter>::matches(&world.archetype_of(entity)) {
            continue;
        }

        // `&mut Position` and `&Velocity`, which is what the metadata recorded.
        let Some(velocity) = world.velocities.get(&entity).copied() else {
            continue;
        };

        if let Some(position) = world.positions.get_mut(&entity) {
            position.x += velocity.x;
            position.y += velocity.y;
        }
    }

    assert_eq!(world.positions[&walker], Position { x: 2.0, y: 3.0 }, "moved");
    assert_eq!(
        world.positions[&corpse],
        Position { x: 100.0, y: 100.0 },
        "excluded by the filter, so untouched"
    );

    // And the function itself is callable, for completeness — it takes a query
    // value, which in this file carries no data.
    movement(Query::new());
    println!("\nmovement ran; the walker moved and the corpse did not.\n");
}

// ===========================================================================
// 12. The whole flow, in one place
// ===========================================================================
//
// ordinary Rust function
//     ↓                       nothing is implemented for it by hand
// function parameter types
//     ↓                       the compiler reads them from the `FnMut` bound
// generic IntoSystem implementation
//     ↓                       chosen by arity; `Params` keeps the impls apart
// each parameter implements SystemParam
//     ↓                       the extension seam: add a param kind, nothing else changes
// Query delegates to QueryData + QueryFilter
//     ↓
// &T           records a read
// &mut T       records a write        — different types, so different impls
// With<T>      records a requirement
// Without<T>   records an exclusion
//     ↓
// RegisteredSystem metadata           — the only thing that reaches run time
//     ↓
// conflict detection                  — write-vs-anything; read-vs-read is fine
//     ↓
// scheduler groups non-conflicting systems
//
// Every arrow above the metadata line happens **while compiling**. Nothing asks
// a value what type it is; the compiler has already chosen which implementations
// to call, and the compiled program simply runs them and inserts `TypeId`s into
// sets. That is what makes this different from reflection: there is no type
// information being consulted at run time, only the results of decisions already
// made.
//
// It is also not overloading. There is one `register_system`, one `IntoSystem`
// trait, one `SystemParam` trait. What varies is which *implementation* of each
// the compiler selects, and it selects by solving the bounds against the types it
// was given.
//
// ---------------------------------------------------------------------------
// Where this could go in Voxel World
// ---------------------------------------------------------------------------
//
// New system parameters are new `SystemParam` implementations, and nothing else:
//
//     Res<T>, ResMut<T>          global values — sketched above
//
//     PropertyRead<P>            read a property from a `PropertyQuery`
//     PropertyWrite<P>           write one
//                                Tracking these in their own sets would let the
//                                conflict rule reason about property access the
//                                same way it reasons about components — and the
//                                schema already knows each property's value kind
//                                and comparability, so the metadata could carry
//                                those too.
//
// And *when* a system runs is a separate question from *what it touches*, which
// is why registration is the natural place to answer it:
//
//     register_on_rota(system, turn)        every Nth tick, one share per turn
//     register_stochastic(system, cadence)  on a drawn interval
//     register_gated(system, gate)          only where a condition holds
//     register_on_condition(system, expr)   when a condition becomes true
//
// Each of those would produce the same `RegisteredSystem` metadata by the same
// route, and differ only in what the scheduler consults before running it. The
// central division is worth stating plainly:
//
//     The function signature says WHAT DATA the system accesses.
//     Registration says WHEN OR WHY the system runs.
//
// Keeping those apart is what lets the conflict rule stay four lines long: it
// never has to know why a system was scheduled, only what it would touch if it
// ran.

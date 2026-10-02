//! Rotas: elements divided into a fixed number of groups of near-equal size.
//!
//! A rota is a roster that says whose turn it is. These structures hold a
//! population and keep it split into `GROUPS` shares that never differ in size
//! by more than one, so a caller can take one share at a time and be sure that
//! every element is covered once per round and that no round carries
//! noticeably more than another.
//!
//! The work they exist for is the kind that has to happen *regularly* but not
//! *simultaneously*: a check that each entity needs once every eight ticks, a
//! revalidation each chunk needs once a minute, a poll each listener needs once
//! a round. Doing all of it on one tick gives a spike; a rota spreads it while
//! keeping the interval exact.
//!
//! ```ignore
//! let mut rota: Rota<EntityId, 8> = Rota::new();
//! rota.insert(entity);
//!
//! // On each tick, one share. Every element is still visited once per eight.
//! for entity in rota.group(tick % 8) {
//!     check(entity);
//! }
//! ```
//!
//! Nothing here mentions ticks: a group is whatever the caller decides a turn
//! is, and the rota only keeps the shares even.
//!
//! # Choosing between them
//!
//! | | Unique elements | Repeats allowed |
//! |---|---|---|
//! | **Order does not matter** | [`Rota`] | [`MultiRota`] |
//! | **Order kept, within a group and overall** | [`OrderedRota`] | [`OrderedMultiRota`] |
//!
//! The unordered pair store each group as a plain list and remove by swapping
//! with its last element, which is constant time and scrambles the order as it
//! goes. The ordered pair keep a sequence number per element: each group reads
//! back in the order its elements were added, and reading the whole rota gives
//! the overall order, so the two orderings agree. That costs a tree lookup
//! where the unordered pair cost nothing, so reach for the ordered ones only
//! when the order is load-bearing.
//!
//! # The balance, and what it costs
//!
//! Every rota keeps its groups within one element of each other at all times.
//! Insertion goes to the emptiest group; removal sometimes has to move one
//! element from the fullest group to the one that shrank, since taking from an
//! already-small group would otherwise open a gap of two.
//!
//! That move is the one surprise worth knowing: **an element does not
//! necessarily stay in the group it first landed in**. Anything that has
//! cached "this element runs on tick three" should ask again rather than
//! remember, through [`Rota::group_of`] or its equivalent.

mod balance;
mod multi_rota;
mod ordered_multi_rota;
mod ordered_rota;
#[allow(clippy::module_inception)]
mod rota;
mod tick_rota;

pub use multi_rota::MultiRota;
pub use ordered_multi_rota::OrderedMultiRota;
pub use ordered_rota::OrderedRota;
pub use rota::Rota;
pub use tick_rota::{TickRota, TickRotaUpdate};

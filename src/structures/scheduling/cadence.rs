//! How often something recurs, and what to do when the clock has run ahead of
//! it.

use crate::random::GeometricRatio;
use crate::units::Ratio;

/// A recurrence described by its mean time to happen.
///
/// Each occurrence waits a geometric number of steps whose mean is
/// [`Cadence::mtth`]: the chance of firing is the same on every step, so the
/// wait has no memory and knowing that something has not happened yet says
/// nothing about when it will. That is the shape most gameplay recurrences
/// want, and the one people reason about most easily — "on average every two
/// hundred ticks" needs no further explanation.
///
/// An *exact* period is a different structure:
/// [`Rota`](crate::structures::collections::Rota) divides a population into
/// even turns and visits each one on schedule, with no randomness at all.
///
/// # Spread
///
/// [`Cadence::spread`] does not change the distribution. It is how far the
/// scheduler may nudge a draw to keep one step from collecting more work than
/// its neighbours: the wait decides roughly where the occurrence lands, the
/// spread decides how much room there is to even out the load once it is there.
/// A spread of zero places every occurrence exactly where its draw fell.
///
/// The nudge shifts the hazard rate by at most `spread` steps, which is nothing
/// against a mean in the hundreds and worth knowing if the two are close.
///
/// # Sharing
///
/// A cadence carries the sampler its mean needs, including the table a rare
/// chance builds once. Register one cadence and give its id to a hundred
/// thousand entries rather than building one apiece.
#[derive(Clone, Debug)]
pub struct Cadence {
    mtth: u64,
    spread: u32,
    backlog: OnBacklog,
    wait: GeometricRatio,
}

impl Cadence {
    /// A recurrence with this mean time to happen, in steps.
    ///
    /// An `mtth` of zero fires on every step; one of `u64::MAX` effectively
    /// never fires, which is how a cadence is switched off without removing its
    /// entries.
    pub fn mtth(mtth: u64) -> Self {
        Self {
            mtth,
            spread: 0,
            backlog: OnBacklog::Coalesce,
            wait: GeometricRatio::one_in(mtth.max(1)),
        }
    }

    /// How far the scheduler may move an occurrence from where its draw fell,
    /// in steps either way, to keep the load even.
    pub fn spread(self, spread: u32) -> Self {
        Self { spread, ..self }
    }

    /// What to do when the clock has run past occurrences that were owed.
    pub fn on_backlog(self, backlog: OnBacklog) -> Self {
        Self { backlog, ..self }
    }

    /// The mean time to happen this cadence was built with, in steps.
    pub const fn mean(&self) -> u64 {
        self.mtth
    }

    /// How far an occurrence may be nudged, in steps either way.
    pub const fn allowed_spread(&self) -> u32 {
        self.spread
    }

    /// What this cadence does with occurrences the clock has passed.
    pub const fn backlog(&self) -> OnBacklog {
        self.backlog
    }

    /// The chance of firing on any one step, which is one in
    /// [`Cadence::mean`].
    pub fn chance(&self) -> Ratio {
        self.wait.chance()
    }

    /// The sampler for the wait, shared by every entry on this cadence.
    pub(crate) const fn wait(&self) -> &GeometricRatio {
        &self.wait
    }
}

/// What a scheduler does with occurrences that fell due while the clock was
/// somewhere else.
///
/// A scheduler advanced fifty steps at once has fifty steps of owed work, and
/// no amount of spreading helps after the fact: the occurrences are already
/// late. What is left is a choice about what "late" means for the work in
/// question.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnBacklog {
    /// One firing that reports how many occurrences it stands for.
    ///
    /// The default, and what most recurring world work wants: a plant that
    /// should have grown three times grows once, by three. The count is on
    /// [`Firing::occurrences`](super::Firing).
    #[default]
    Coalesce,
    /// One firing per owed occurrence, all delivered at once.
    ///
    /// For work where each occurrence has its own effect and none may be
    /// skipped. A long skip delivers a proportionally long list.
    FireAll,
    /// At most this many firings per step, with the rest still owed.
    ///
    /// Smooths a backlog out over the steps that follow instead of paying for
    /// it at once. The owed occurrences keep their place in the queue, so the
    /// work drains rather than disappearing, and an entry that is permanently
    /// behind stays behind.
    Cap(u32),
}

//! Kinematic quantities: what a vector of [`Fixed`] actually *means*.
//!
//! # The problem
//!
//! ```ignore
//! position += velocity * Fixed::from_f64(delta);
//! ```
//!
//! Every value there is a `Vector3<Fixed>` or a `Fixed`, so nothing stops the same line
//! being written with an acceleration, a force, a colour or a tick count in place of the
//! velocity. It compiles, it runs, and the world is wrong by whatever factor the mistake
//! happened to be. The representation carries no meaning, so the compiler can check
//! none.
//!
//! These wrappers put the meaning back:
//!
//! ```text
//! Displacement   how far, and which way
//! Velocity       how fast, and which way          displacement / second
//! Acceleration   how velocity changes             velocity / second
//! Force          what produces acceleration       mass × acceleration
//! Mass           how much there is to push
//! ```
//!
//! and [`PrecisePosition3`] — which already existed — is *where*.
//!
//! # Dimensions
//!
//! Two, three and four components, from one macro per dimension. There is no four-
//! dimensional *position* type in the engine, so the 4D quantities stand on their own
//! for whatever wants them — a spacetime offset, a four-channel field, a homogeneous
//! coordinate — while [`PrecisePosition3`] is the only thing a displacement can move.
//!
//! # Units
//!
//! One set, stated once, with no dimensional-analysis framework behind it:
//!
//! | quantity | unit |
//! |---|---|
//! | position, displacement | world units |
//! | velocity | world units per second |
//! | acceleration | world units per second squared |
//! | mass | mass units |
//! | force | mass units × world units per second squared |
//!
//! A world unit is whatever [`PreciseUnits`](super::precise::PreciseUnits) counts.
//! Seconds are **real** seconds: see the second-to-tick note below.
//!
//! # What the compiler now refuses
//!
//! ```compile_fail
//! use voxel_world::spatial::kinematics::{Acceleration3, Velocity3};
//! use voxel_world::math::Fixed;
//!
//! let velocity = Velocity3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
//! let acceleration = Acceleration3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
//!
//! // Different quantities do not add, however alike they look underneath.
//! let _ = velocity + acceleration;
//! ```
//!
//! ```compile_fail
//! use voxel_world::spatial::kinematics::{Force3, Velocity3};
//! use voxel_world::math::Fixed;
//!
//! let velocity = Velocity3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
//! let force = Force3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
//!
//! let _ = force + velocity;
//! ```
//!
//! ```compile_fail
//! use voxel_world::spatial::kinematics::Velocity3;
//! use voxel_world::math::Fixed;
//! use voxel_world::time::TickDuration;
//!
//! let velocity = Velocity3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
//!
//! // A count of ticks is not a number of seconds. Go through TickRate.
//! let _ = velocity * TickDuration::new(10);
//! ```
//!
//! ```compile_fail
//! use voxel_world::spatial::kinematics::Velocity3;
//! use voxel_world::math::Fixed;
//! use voxel_world::units::frame::TickAlpha;
//!
//! let velocity = Velocity3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
//!
//! // An interpolation ratio is dimensionless, not a duration.
//! let _ = velocity * TickAlpha::HALF;
//! ```
//!
//! ```compile_fail
//! use voxel_world::spatial::kinematics::Acceleration3;
//! use voxel_world::math::Fixed;
//! use voxel_world::units::frame::TickAlpha;
//!
//! let acceleration = Acceleration3::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
//!
//! let _ = acceleration * TickAlpha::HALF;
//! ```
//!
//! Four dimensions keep the same guarantees, since they come from the same macro:
//!
//! ```compile_fail
//! use voxel_world::spatial::kinematics::{Acceleration4, Velocity4};
//! use voxel_world::math::Fixed;
//!
//! let velocity = Velocity4::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
//! let acceleration = Acceleration4::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
//!
//! let _ = velocity + acceleration;
//! ```
//!
//! ```compile_fail
//! use voxel_world::spatial::kinematics::Velocity4;
//! use voxel_world::math::Fixed;
//! use voxel_world::units::frame::TickAlpha;
//!
//! let velocity = Velocity4::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
//!
//! let _ = velocity * TickAlpha::HALF;
//! ```
//!
//! A frame delta and an update delta are not each other, however alike their contents:
//!
//! ```compile_fail
//! use voxel_world::time::UpdateDelta;
//! use voxel_world::units::frame::FrameDelta;
//!
//! let update: UpdateDelta = UpdateDelta::ZERO;
//! let _: FrameDelta = update;
//! ```
//!
//! # The two durations that do work
//!
//! ```
//! use voxel_world::math::Fixed;
//! use voxel_world::spatial::kinematics::Velocity3;
//! use voxel_world::time::{Seconds, TickDuration, TickRate, UpdateDelta};
//! use voxel_world::units::frame::FrameDelta;
//!
//! let rate = TickRate::new(60).unwrap();
//! let velocity = Velocity3::new(Fixed::from_integer(10), Fixed::ZERO, Fixed::ZERO);
//!
//! // A frame's measured delta, for presentation.
//! let drawn = velocity * FrameDelta::from_millis(500).unwrap();
//!
//! // An authoritative update's simulation time.
//! let stepped = velocity * UpdateDelta::from_ticks(TickDuration::new(30), rate);
//!
//! // A general real duration, for anything that is neither.
//! let plain = velocity * Seconds::from_millis(500).unwrap();
//!
//! assert_eq!(drawn, plain, "the same duration, named two ways");
//! assert_eq!(stepped, plain, "and a third, reached by counting ticks");
//! ```
//!
//! # Ticks are not seconds
//!
//! A velocity here is *per second*, and a [`TickDuration`](crate::time::TickDuration)
//! is a count of ticks — which means a number of seconds only once a
//! [`TickRate`](crate::time::TickRate) says so. So the conversion is explicit:
//!
//! ```
//! use voxel_world::math::Fixed;
//! use voxel_world::spatial::kinematics::Velocity3;
//! use voxel_world::time::{TickDuration, TickRate};
//!
//! let rate = TickRate::new(50).expect("positive");
//! let velocity = Velocity3::new(Fixed::from_integer(10), Fixed::ZERO, Fixed::ZERO);
//!
//! // One second of ticks at fifty a second.
//! let over = rate.seconds_in(TickDuration::new(50));
//! let moved = velocity * over;
//!
//! assert_eq!(moved.as_vector().x, Fixed::from_integer(10));
//! ```
//!
//! # Simulation against presentation
//!
//! Two paths reach the same integration, and which one a piece of code is on decides
//! whether it may change the world:
//!
//! ```text
//! AUTHORITATIVE SIMULATION
//!     TickDuration                 how many steps
//!         ↓  through TickRate      ← the only place ticks become seconds
//!     UpdateDelta                  how much simulation time those steps are
//!         ↓
//!     velocity * delta             identical on every machine
//!
//! PRESENTATION
//!     FrameDelta                   what the platform measured for this frame
//!         ↓
//!     velocity * delta             varies with the machine, and may not change the world
//! ```
//!
//! [`Seconds`] sits under both as the general physical duration,
//! and the operators accept it directly for anything that is neither — a timeout, an
//! animation length, a measured interval.
//!
//! Both end in fixed-point seconds, and the operators accept either. The difference is
//! **origin**: a [`Seconds`] derived from a
//! [`TickRate`](crate::time::TickRate) is a fact about the simulation, where a
//! [`FrameDelta`] is a fact about the hardware. Driving the world from the second would
//! make its contents depend on the frame rate, and two players would diverge.
//!
//! So ordinary frame work — a camera easing, a particle, an interpolated draw — uses
//! `frame.delta()`, and an authoritative periodic system uses `update.delta()` from a
//! [`TickRota`](crate::structures::collections::rota::TickRota), which
//! works out the elapsed ticks for itself.
//!
//! # Overflow
//!
//! The operators panic on overflow, exactly as [`Fixed`]'s own do, because an
//! overflowing displacement is a bug rather than a value. Every one has a
//! `checked_` counterpart returning [`Option`] for code that would rather decide.

use crate::math::{Fixed, Vector2, Vector3, Vector4};
use crate::spatial::precise::PrecisePosition3;
use crate::units::frame::FrameDelta;
use crate::units::time::{Seconds, UpdateDelta};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// How much there is to push: a non-negative scalar amount of mass.
///
/// # Question
///
/// "How hard is this to accelerate?"
///
/// # Zero is allowed to exist but not to divide
///
/// A massless thing is a reasonable thing to describe, so [`Mass::ZERO`] exists. What
/// is not reasonable is `force / 0`, which has no answer — so
/// [`Force3::checked_divide`] reports it rather than producing an infinity that
/// travels into a position before anyone notices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Mass(Fixed);

impl Mass {
    /// No mass at all.
    pub const ZERO: Self = Self(Fixed::ZERO);

    /// One mass unit.
    pub const ONE: Self = Self(Fixed::ONE);

    /// A mass, or `None` if negative.
    pub const fn new(amount: Fixed) -> Option<Self> {
        if amount.to_bits() < 0 {
            return None;
        }

        Some(Self(amount))
    }

    /// A whole number of mass units, or `None` if negative or too large.
    pub const fn from_integer(amount: i64) -> Option<Self> {
        if amount < 0 {
            return None;
        }

        match Fixed::from_integer_i128(amount as i128) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// The amount, as the engine's scalar.
    pub const fn amount(self) -> Fixed {
        self.0
    }

    /// Whether there is no mass here.
    pub const fn is_zero(self) -> bool {
        self.0.to_bits() == 0
    }
}

impl fmt::Display for Mass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} mass", self.0)
    }
}

/// Writes one dimension's worth of kinematic quantities and the relations between them.
///
/// One invocation per dimension, so the four types and every operator between them are
/// stated once rather than four times — and a new dimension is one line.
macro_rules! kinematics {
    (
        $vector:ident, $displacement:ident, $velocity:ident, $acceleration:ident,
        $force:ident, $dimensions:literal
    ) => {
        #[doc = concat!("A change in position in ", $dimensions, " dimensions, in world units.")]
        ///
        /// What a velocity integrated over a duration produces, and the only thing a
        /// position can be moved by.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $displacement($vector<Fixed>);

        #[doc = concat!("How fast and in which direction, in ", $dimensions, " dimensions: world units per second.")]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $velocity($vector<Fixed>);

        #[doc = concat!("How a velocity changes, in ", $dimensions, " dimensions: world units per second squared.")]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $acceleration($vector<Fixed>);

        #[doc = concat!("What produces acceleration given mass, in ", $dimensions, " dimensions.")]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $force($vector<Fixed>);

        kinematic_common!($displacement, $vector);
        kinematic_common!($velocity, $vector);
        kinematic_common!($acceleration, $vector);
        kinematic_common!($force, $vector);

        /// A velocity over a real duration is a displacement.
        impl Mul<Seconds> for $velocity {
            type Output = $displacement;
            fn mul(self, over: Seconds) -> $displacement {
                $displacement(self.0 * over.fixed())
            }
        }

        /// An acceleration over a real duration is a change in velocity.
        impl Mul<Seconds> for $acceleration {
            type Output = $velocity;
            fn mul(self, over: Seconds) -> $velocity {
                $velocity(self.0 * over.fixed())
            }
        }

        /// A velocity over a frame's measured time is a displacement.
        ///
        /// The form frame code is written in: `position += velocity * frame.delta()`.
        impl Mul<FrameDelta> for $velocity {
            type Output = $displacement;
            fn mul(self, over: FrameDelta) -> $displacement {
                $displacement(self.0 * over.fixed())
            }
        }

        /// An acceleration over a frame's measured time is a change in velocity.
        impl Mul<FrameDelta> for $acceleration {
            type Output = $velocity;
            fn mul(self, over: FrameDelta) -> $velocity {
                $velocity(self.0 * over.fixed())
            }
        }

        /// A velocity over an update's simulation time is a displacement.
        ///
        /// The authoritative form: `position += velocity * update.delta()`.
        impl Mul<UpdateDelta> for $velocity {
            type Output = $displacement;
            fn mul(self, over: UpdateDelta) -> $displacement {
                $displacement(self.0 * over.fixed())
            }
        }

        /// An acceleration over an update's simulation time is a change in velocity.
        impl Mul<UpdateDelta> for $acceleration {
            type Output = $velocity;
            fn mul(self, over: UpdateDelta) -> $velocity {
                $velocity(self.0 * over.fixed())
            }
        }

        /// Mass times acceleration is force.
        impl Mul<Mass> for $acceleration {
            type Output = $force;
            fn mul(self, mass: Mass) -> $force {
                $force(self.0 * mass.amount())
            }
        }

        /// Mass times acceleration is force, the other way round.
        impl Mul<$acceleration> for Mass {
            type Output = $force;
            fn mul(self, acceleration: $acceleration) -> $force {
                acceleration * self
            }
        }

        impl $velocity {
            /// The displacement over `over`, or `None` on overflow.
            pub fn checked_over(self, over: Seconds) -> Option<$displacement> {
                self.0.checked_scale(over.fixed()).map($displacement)
            }

            /// The displacement over a frame's measured time, or `None` on overflow.
            pub fn checked_over_delta(self, over: FrameDelta) -> Option<$displacement> {
                self.0.checked_scale(over.fixed()).map($displacement)
            }

            /// The displacement over an update's simulation time, or `None` on
            /// overflow.
            pub fn checked_over_update(self, over: UpdateDelta) -> Option<$displacement> {
                self.0.checked_scale(over.fixed()).map($displacement)
            }
        }

        impl $acceleration {
            /// The change in velocity over `over`, or `None` on overflow.
            pub fn checked_over(self, over: Seconds) -> Option<$velocity> {
                self.0.checked_scale(over.fixed()).map($velocity)
            }

            /// The change in velocity over a frame's measured time, or `None` on
            /// overflow.
            pub fn checked_over_delta(self, over: FrameDelta) -> Option<$velocity> {
                self.0.checked_scale(over.fixed()).map($velocity)
            }

            /// The change in velocity over an update's simulation time, or `None` on
            /// overflow.
            pub fn checked_over_update(self, over: UpdateDelta) -> Option<$velocity> {
                self.0.checked_scale(over.fixed()).map($velocity)
            }

            /// The displacement from starting at `initial` and accelerating for `over`.
            ///
            /// `initial × t + acceleration × t² / 2`, the constant-acceleration result,
            /// in one expression rather than a loop. `None` on overflow.
            pub fn checked_displacement(
                self,
                initial: $velocity,
                over: Seconds,
            ) -> Option<$displacement> {
                let seconds: Fixed = over.fixed();
                let from_speed: $vector<Fixed> = initial.0.checked_scale(seconds)?;
                let half_square: Fixed = seconds.checked_mul(seconds)?.checked_div(Fixed::from_integer(2))?;
                let from_acceleration: $vector<Fixed> = self.0.checked_scale(half_square)?;

                from_speed.checked_add(from_acceleration).map($displacement)
            }

            /// The displacement over a frame's measured time; see
            #[doc = concat!("[`", stringify!($acceleration), "::checked_displacement`].")]
            pub fn checked_displacement_delta(
                self,
                initial: $velocity,
                over: FrameDelta,
            ) -> Option<$displacement> {
                self.checked_displacement(initial, over.seconds())
            }

            /// The displacement over an update's simulation time; see
            #[doc = concat!("[`", stringify!($acceleration), "::checked_displacement`].")]
            pub fn checked_displacement_update(
                self,
                initial: $velocity,
                over: UpdateDelta,
            ) -> Option<$displacement> {
                self.checked_displacement(initial, over.seconds())
            }
        }

        impl $force {
            /// The acceleration this force gives `mass`, or `None` for zero mass or on
            /// overflow.
            ///
            /// Zero mass is reported rather than divided by: there is no acceleration
            /// that answers "what does a push do to nothing?".
            pub fn checked_divide(self, mass: Mass) -> Option<$acceleration> {
                if mass.is_zero() {
                    return None;
                }

                // Divided component by component rather than scaled by a reciprocal: a
                // reciprocal rounds, so dividing by three used to give 1.9999999995
                // where it now gives exactly 2.
                self.0.checked_divide(mass.amount()).map($acceleration)
            }
        }

        /// Force divided by mass is acceleration.
        ///
        /// # Panics
        ///
        /// On zero mass, where there is no answer. Use [`checked_divide`](
        #[doc = concat!("`", stringify!($force), "::checked_divide`")]
        /// ) where the mass might be zero.
        impl Div<Mass> for $force {
            type Output = $acceleration;
            fn div(self, mass: Mass) -> $acceleration {
                self.checked_divide(mass)
                    .expect("a force divided by zero mass has no acceleration")
            }
        }
    };
}

/// The part every kinematic vector shares: construction, access, and arithmetic with
/// its own kind.
macro_rules! kinematic_common {
    ($name:ident, $vector:ident) => {
        impl $name {
            /// Nothing at all.
            pub const ZERO: Self = Self($vector::ZERO_FIXED);

            /// From an existing vector, declaring what it means.
            pub const fn from_vector(vector: $vector<Fixed>) -> Self {
                Self(vector)
            }

            /// The underlying vector, for mathematics this type does not cover.
            ///
            /// Deliberately a method rather than a public field: reaching for it should
            /// read as a decision to leave the typed world, not as the default way to
            /// use one of these.
            pub const fn as_vector(self) -> $vector<Fixed> {
                self.0
            }

            /// The underlying vector, consuming this.
            pub const fn into_vector(self) -> $vector<Fixed> {
                self.0
            }

            /// Both together, or `None` on overflow.
            pub fn checked_add(self, other: Self) -> Option<Self> {
                self.0.checked_add(other.0).map(Self)
            }

            /// The difference, or `None` on overflow.
            pub fn checked_sub(self, other: Self) -> Option<Self> {
                self.0.checked_sub(other.0).map(Self)
            }

            /// Scaled by a dimensionless factor, or `None` on overflow.
            pub fn checked_scale(self, factor: Fixed) -> Option<Self> {
                self.0.checked_scale(factor).map(Self)
            }
        }

        /// Two of the same quantity add.
        impl Add for $name {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                *self = *self + other;
            }
        }

        /// Two of the same quantity subtract.
        impl Sub for $name {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                Self(self.0 - other.0)
            }
        }

        impl SubAssign for $name {
            fn sub_assign(&mut self, other: Self) {
                *self = *self - other;
            }
        }

        /// The same quantity the other way.
        impl Neg for $name {
            type Output = Self;
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        /// Scaled by a dimensionless factor, which leaves the quantity what it was.
        impl Mul<Fixed> for $name {
            type Output = Self;
            fn mul(self, factor: Fixed) -> Self {
                Self(self.0 * factor)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{:?}", self.0)
            }
        }
    };
}

/// A zero vector of `Fixed`, which the generic vector types cannot provide as a
/// constant because their own `ZERO` is written for floats.
trait ZeroFixed {
    const ZERO_FIXED: Self;
}

impl ZeroFixed for Vector2<Fixed> {
    const ZERO_FIXED: Self = Self {
        x: Fixed::ZERO,
        y: Fixed::ZERO,
    };
}

impl ZeroFixed for Vector3<Fixed> {
    const ZERO_FIXED: Self = Self {
        x: Fixed::ZERO,
        y: Fixed::ZERO,
        z: Fixed::ZERO,
    };
}

impl ZeroFixed for Vector4<Fixed> {
    const ZERO_FIXED: Self = Self {
        x: Fixed::ZERO,
        y: Fixed::ZERO,
        z: Fixed::ZERO,
        w: Fixed::ZERO,
    };
}

kinematics!(Vector2, Displacement2, Velocity2, Acceleration2, Force2, "two");
kinematics!(Vector3, Displacement3, Velocity3, Acceleration3, Force3, "three");
kinematics!(Vector4, Displacement4, Velocity4, Acceleration4, Force4, "four");

impl Velocity2 {
    /// A velocity from its components, in world units per second.
    pub const fn new(x: Fixed, y: Fixed) -> Self {
        Self(Vector2 { x, y })
    }
}

impl Velocity3 {
    /// A velocity from its components, in world units per second.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self(Vector3 { x, y, z })
    }
}

impl Acceleration2 {
    /// An acceleration from its components, in world units per second squared.
    pub const fn new(x: Fixed, y: Fixed) -> Self {
        Self(Vector2 { x, y })
    }
}

impl Acceleration3 {
    /// An acceleration from its components, in world units per second squared.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self(Vector3 { x, y, z })
    }
}

impl Displacement2 {
    /// A displacement from its components, in world units.
    pub const fn new(x: Fixed, y: Fixed) -> Self {
        Self(Vector2 { x, y })
    }
}

impl Displacement3 {
    /// A displacement from its components, in world units.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self(Vector3 { x, y, z })
    }
}

impl Force2 {
    /// A force from its components.
    pub const fn new(x: Fixed, y: Fixed) -> Self {
        Self(Vector2 { x, y })
    }
}

impl Force3 {
    /// A force from its components.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self(Vector3 { x, y, z })
    }
}

impl Velocity4 {
    /// A velocity from its components, in world units per second.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Self {
        Self(Vector4 { x, y, z, w })
    }
}

impl Acceleration4 {
    /// An acceleration from its components, in world units per second squared.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Self {
        Self(Vector4 { x, y, z, w })
    }
}

impl Displacement4 {
    /// A displacement from its components, in world units.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Self {
        Self(Vector4 { x, y, z, w })
    }
}

impl Force4 {
    /// A force from its components.
    pub const fn new(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Self {
        Self(Vector4 { x, y, z, w })
    }
}

// ---------------------------------------------------------------------------
// Where: the position that already existed
// ---------------------------------------------------------------------------

/// A position moved by a displacement is a position.
///
/// # Panics
///
/// On overflow, as the other operators here do. [`PrecisePosition3::checked_offset`]
/// takes a bare vector and reports instead.
impl Add<Displacement3> for PrecisePosition3 {
    type Output = Self;
    fn add(self, displacement: Displacement3) -> Self {
        self.checked_offset(displacement.as_vector())
            .expect("a position moved past the representable world")
    }
}

impl AddAssign<Displacement3> for PrecisePosition3 {
    fn add_assign(&mut self, displacement: Displacement3) {
        *self = *self + displacement;
    }
}

/// A position moved back by a displacement.
impl Sub<Displacement3> for PrecisePosition3 {
    type Output = Self;
    fn sub(self, displacement: Displacement3) -> Self {
        self + (-displacement)
    }
}

impl SubAssign<Displacement3> for PrecisePosition3 {
    fn sub_assign(&mut self, displacement: Displacement3) {
        *self = *self - displacement;
    }
}

impl PrecisePosition3 {
    /// The displacement from `earlier` to here, or `None` on overflow.
    ///
    /// Two positions subtract to the distance between them, which is a displacement and
    /// not another position — the same distinction [`Tick`](crate::time::Tick) and
    /// [`TickDuration`](crate::time::TickDuration) draw in time.
    pub fn displacement_from(self, earlier: Self) -> Option<Displacement3> {
        self.checked_difference(earlier).map(Displacement3::from_vector)
    }

    /// This position after moving at `velocity` for `over`, or `None` on overflow.
    ///
    /// # Example
    ///
    /// ```
    /// use voxel_world::math::Fixed;
    /// use voxel_world::spatial::kinematics::{Displacement3, Velocity3};
    /// use voxel_world::spatial::precise::PrecisePosition3;
    /// use voxel_world::time::Seconds;
    ///
    /// let start = PrecisePosition3::new(Fixed::ZERO, Fixed::ZERO, Fixed::ZERO);
    /// let velocity = Velocity3::new(Fixed::from_integer(10), Fixed::ZERO, Fixed::ZERO);
    ///
    /// let moved = start.moved_by(velocity, Seconds::from_seconds(2).unwrap()).unwrap();
    ///
    /// // Ten units a second for two seconds is twenty units.
    /// assert_eq!(
    ///     moved.displacement_from(start),
    ///     Some(Displacement3::new(Fixed::from_integer(20), Fixed::ZERO, Fixed::ZERO)),
    /// );
    /// ```
    pub fn moved_by(self, velocity: Velocity3, over: Seconds) -> Option<Self> {
        let displacement: Displacement3 = velocity.checked_over(over)?;

        self.checked_offset(displacement.as_vector())
    }
}

impl From<Vector3<Fixed>> for Displacement3 {
    fn from(vector: Vector3<Fixed>) -> Self {
        Self::from_vector(vector)
    }
}

//! Simulation time and frame timing.
//!
//! Quantities are defined in [`crate::units`]. This module collects them with
//! the frame clock so callers can find the full time API in one place.
//!
//! ```
//! use voxel_world::time::{FrameClock, FrameDelta, TickRate};
//! let mut clock = FrameClock::new(TickRate::new(60).unwrap());
//! clock.begin_frame(FrameDelta::from_millis(17).unwrap());
//! assert_eq!(clock.ready_ticks().count(), 1);
//! ```

pub use crate::units::frame::{
    DeltaOutOfRange, Frame, FrameClock, FrameDelta, FrameIndex, TickAlpha,
};
pub use crate::units::time::{Seconds, Tick, TickDuration, TickRate, UpdateDelta};

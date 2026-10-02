//! A window, a Vulkan triangle, and a simulation that ticks at a fixed rate.
//!
//! Run it with `cargo run`. Escape closes the window.
//!
//! # Where to experiment
//!
//! - [`Playground::simulation_tick`] is the fixed-rate method. It runs exactly
//!   [`TICKS_PER_SECOND`] times a second, and is where world state belongs:
//!   advancing schedulers and rotas, draining gates and condition indices.
//! - [`renderer`] holds the Vulkan setup; `Renderer::record` writes the
//!   commands for one frame.
//! - `shaders/` holds the GLSL, recompiled by `build.rs` on every build.
//!
//! # Shaders
//!
//! Vulkan has no shader language of its own: it consumes **SPIR-V**, a binary
//! format, and will not take text. GLSL is how that SPIR-V is written, and
//! `glslc` does the translating:
//!
//! ```text
//! shaders/triangle.vert  (GLSL)  ->  glslc  ->  triangle.vert.spv  (SPIR-V)
//! ```
//!
//! HLSL and Slang compile to SPIR-V too, if either suits better.
//!
//! # The timing
//!
//! Simulation time follows a fixed timeline rather than the clock: each tick
//! adds one interval to the time the *next* one is due, so a slow tick is made
//! up rather than pushing everything after it later for ever. Several overdue
//! ticks run one after another to catch up, and no more than
//! [`MAX_CATCH_UP_TICKS`] of them in one pass, so falling behind cannot feed
//! itself into a spiral.
//!
//! When even that many is not enough, the timeline is resynchronised and the
//! backlog is dropped. That suits a client, where being current matters more
//! than having run every tick; a server replaying a world would rather keep
//! them all, and would raise or remove the cap instead.
//!
//! Drawing is not tied to the tick. The event loop is told to wake when the
//! next tick is due, so waiting costs nothing and the window stays responsive —
//! the same timeline as sleeping the thread, without going deaf to events while
//! asleep.

mod renderer;

use renderer::Renderer;
use std::sync::Arc;
use std::time::{Duration, Instant};
use voxel_world::math::Vector3;
use voxel_world::random::seed::Seed;
use voxel_world::spatial::VoxelPosition3;
use voxel_world::time::{Seconds, Tick, TickDuration, TickRate};
use voxel_world::units::frame::{Frame, FrameClock, FrameDelta};
use voxel_world::world::{ChunkGenerator, VoxelType, VoxelTypeIndex};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// How many times a second [`Playground::simulation_tick`] runs.
const TICKS_PER_SECOND: u32 = 60;

/// The most overdue ticks to run in one pass before giving up and
/// resynchronising.
///
/// Without a cap, a long stall leaves so many ticks owed that running them
/// takes longer than the stall did, which owes more still.
///
/// A [`TickDuration`] rather than a count, so it can be compared against the
/// ticks actually caught up without either side being a bare number.
const MAX_CATCH_UP_TICKS: TickDuration = TickDuration::new(5);

fn main() {
    describe_world();

    let event_loop: EventLoop<()> = EventLoop::new().expect("an event loop");

    // Wait for the next tick rather than spinning; `about_to_wait` sets the
    // time each pass.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut playground = Playground::new();

    event_loop
        .run_app(&mut playground)
        .expect("the event loop ran");
}

/// Generates one chunk and says what is at its origin.
///
/// What `main` used to be, kept as a sign that the world half still works while
/// the window half is being built.
fn describe_world() {
    let mut registry = VoxelTypeIndex::new();
    let air = registry.add_voxel_type(VoxelType::new("air")).unwrap();
    let stone = registry.add_voxel_type(VoxelType::new("stone")).unwrap();

    let generator = ChunkGenerator::new(
        Seed::from_text("example world"),
        &registry,
        &[(air, 1.0), (stone, 3.0)],
    )
    .unwrap();

    let origin = VoxelPosition3::new(Vector3::new(0, 0, 0));
    let chunk = generator.generate(origin).unwrap();
    let material = registry
        .get_voxel_type(chunk.voxel(origin).unwrap())
        .unwrap();

    println!(
        "Generated {} voxels; the origin contains {}.",
        chunk.voxels().len(),
        material.name()
    );
}

struct Playground {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,

    /// Turns measured frame time into whole simulation ticks.
    ///
    /// Holds the rate, the accumulator, the current [`Tick`] and the frame
    /// index, so none of that is tracked by hand here. It deliberately holds no
    /// policy: the catch-up cap below is this loop's decision, not the clock's.
    clock: FrameClock,
    /// When the last frame was measured — the only place the real clock is read.
    last_frame: Instant,

    /// How many ticks were dropped to catch up. A duration, since it is an
    /// amount of time and not a moment in it.
    dropped: TickDuration,
}

impl Playground {
    fn new() -> Self {
        let rate: TickRate = TickRate::new(TICKS_PER_SECOND).expect("a positive tick rate");

        Self {
            window: None,
            renderer: None,
            clock: FrameClock::new(rate),
            last_frame: Instant::now(),
            dropped: TickDuration::ZERO,
        }
    }

    /// One step of the world. **This is the method to fill in.**
    ///
    /// It runs at a fixed rate, so anything here can assume exactly one
    /// interval has passed since the last call, whatever the frame rate is
    /// doing.
    fn simulation_tick(&mut self, now: Tick) {
        // A sign of life once a second, which is also a reminder that the rate
        // is fixed: this stays at one line a second whatever the window does.
        //
        // `is_every` and `seconds_at` are what the tick types are for — the
        // schedule and the clock conversion are asked for by name rather than
        // rebuilt out of a modulo and a division that could each be wrong.
        if now.is_every(TickDuration::new(u64::from(TICKS_PER_SECOND))) {
            let elapsed: Seconds = self.clock.rate().seconds_at(now);

            println!(
                "{now} ({:.0}s, {} dropped)",
                elapsed.to_f64(),
                self.dropped,
            );
        }
    }

    /// Measures the frame, runs the ticks it paid for, and says when to wake.
    fn run_due_ticks(&mut self) -> Instant {
        // The one place the real clock is read. A `Duration` holds integer
        // seconds and nanoseconds, and `FrameDelta::from_duration` scales those
        // into `Fixed` directly — no `as_secs_f64`, so nothing is rounded twice.
        let measured: Instant = Instant::now();
        let delta: FrameDelta = FrameDelta::from_duration(measured.duration_since(self.last_frame))
            .unwrap_or(FrameDelta::ZERO);

        self.last_frame = measured;

        let frame: Frame = self.clock.begin_frame(delta);

        // The clock says how many ticks the elapsed time paid for; the cap is
        // this loop's policy, which is why the clock knows nothing about it.
        let mut caught_up: TickDuration = TickDuration::ZERO;

        while caught_up < MAX_CATCH_UP_TICKS {
            let Some(now) = self.clock.take_tick() else {
                break;
            };

            self.simulation_tick(now);
            caught_up += TickDuration::ONE;
        }

        // Still behind after a full pass, so the backlog is abandoned rather
        // than chased. Counted, because silently losing time should show up in a
        // log rather than as a mystery.
        self.dropped += self.clock.discard_backlog();

        // Whatever is left in the accumulator is how far into the next tick the
        // clock already is, so the next one falls due the rest of a tick away.
        // `frame.tick_alpha()` is that same leftover as a fraction, which is what
        // a renderer interpolates the world's last two states by.
        let remaining: Seconds = self.clock.tick_length() - self.clock.accumulated();
        let _ = frame.tick_alpha();

        // An `f64` here is correct: this is the boundary out to the operating
        // system, and a `Duration` is what it asked for.
        measured + Duration::from_secs_f64(remaining.to_f64())
    }
}

impl ApplicationHandler for Playground {
    /// Called once at start-up, and again if the platform ever takes the window
    /// away and gives it back.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("voxel-world playground")
            .with_inner_size(winit::dpi::LogicalSize::new(960.0, 640.0));

        let window: Arc<Window> = Arc::new(event_loop.create_window(attributes).expect("a window"));

        self.renderer = Some(Renderer::new(Arc::clone(&window)));
        self.window = Some(window);

        // Start the timeline now rather than whenever the struct was made, so
        // start-up does not count as a stall to be caught up.
        self.last_frame = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }

            WindowEvent::Resized(_) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resized();
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.draw();
                }
            }

            _ => {}
        }
    }

    /// Runs after the events waiting have been handled, which is where the
    /// timing lives.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let next: Instant = self.run_due_ticks();

        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }

        // Sleeping the thread would work and keep the same timeline, but the
        // window would stop answering the system while asleep. Asking the event
        // loop to wake at the same moment keeps both.
        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
    }

    /// Called as the loop ends, which is where Vulkan is given back.
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.destroy();
        }

        println!(
            "ran {} ticks over {} frames, dropped {}",
            self.clock.now().ticks_since(Tick::ORIGIN),
            self.clock.frames_presented(),
            self.dropped,
        );
    }
}

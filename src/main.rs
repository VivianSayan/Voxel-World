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
const MAX_CATCH_UP_TICKS: usize = 5;

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

    /// How long one tick lasts.
    tick_interval: Duration,
    /// When the next tick is due. Advanced by exactly one interval per tick,
    /// never reset to the clock, so the timeline does not drift.
    next_tick: Instant,

    /// How many ticks have run, and how many were dropped to catch up.
    ticks: u64,
    dropped: u64,
}

impl Playground {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            tick_interval: Duration::from_secs_f64(1.0 / f64::from(TICKS_PER_SECOND)),
            next_tick: Instant::now(),
            ticks: 0,
            dropped: 0,
        }
    }

    /// One step of the world. **This is the method to fill in.**
    ///
    /// It runs at a fixed rate, so anything here can assume exactly one
    /// interval has passed since the last call, whatever the frame rate is
    /// doing.
    fn simulation_tick(&mut self) {
        self.ticks += 1;

        // A sign of life once a second, which is also a reminder that the rate
        // is fixed: this stays at one line a second whatever the window does.
        if self.ticks.is_multiple_of(u64::from(TICKS_PER_SECOND)) {
            let seconds: u64 = self.ticks / u64::from(TICKS_PER_SECOND);
            println!("tick {} ({seconds}s, {} dropped)", self.ticks, self.dropped);
        }
    }

    /// Runs whatever ticks are due, and says when the next one is.
    fn run_due_ticks(&mut self) -> Instant {
        let mut caught_up: usize = 0;

        while Instant::now() >= self.next_tick && caught_up < MAX_CATCH_UP_TICKS {
            self.simulation_tick();

            // The intended timeline, not the clock: a tick that ran late does
            // not push the ones after it late as well.
            self.next_tick += self.tick_interval;
            caught_up += 1;
        }

        // Still behind after a full pass, so the backlog is dropped rather than
        // chased. Counted, because silently losing time is the sort of thing
        // that should show up in a log rather than as a mystery.
        if caught_up == MAX_CATCH_UP_TICKS && Instant::now() >= self.next_tick {
            let behind: Duration = Instant::now().duration_since(self.next_tick);
            self.dropped += (behind.as_secs_f64() / self.tick_interval.as_secs_f64()) as u64;
            self.next_tick = Instant::now() + self.tick_interval;
        }

        self.next_tick
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
        self.next_tick = Instant::now();
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

        println!("ran {} ticks, dropped {}", self.ticks, self.dropped);
    }
}

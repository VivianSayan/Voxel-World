//! Compiles the GLSL in `shaders/` to SPIR-V beside the build output.
//!
//! Edit a shader and `cargo run` picks it up; there is no separate step.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out: PathBuf = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));

    for shader in ["triangle.vert", "triangle.frag"] {
        let source: PathBuf = Path::new("shaders").join(shader);
        let target: PathBuf = out.join(format!("{shader}.spv"));

        println!("cargo:rerun-if-changed={}", source.display());

        let result = Command::new("glslc")
            .arg(&source)
            .arg("-o")
            .arg(&target)
            .output();

        match result {
            Ok(output) if output.status.success() => {}
            Ok(output) => panic!(
                "glslc rejected {}:\n{}",
                source.display(),
                String::from_utf8_lossy(&output.stderr)
            ),
            Err(error) => panic!(
                "could not run glslc ({error}). It comes with shaderc:\n    \
                 brew install shaderc"
            ),
        }
    }
}

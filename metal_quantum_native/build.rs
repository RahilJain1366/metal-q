use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    // Workspace root is one level above the crate manifest dir.
    let workspace_root = PathBuf::from(&manifest_dir)
        .parent()
        .expect("crate must be inside a workspace")
        .to_path_buf();

    let shaders = ["gates", "two_qubit", "gradient"];
    let mut air_files: Vec<String> = Vec::new();

    for shader in &shaders {
        let src = workspace_root.join(format!("shaders/{shader}.metal"));
        let air = format!("{out}/{shader}.air");

        println!("cargo:rerun-if-changed={}", src.display());

        if !src.exists() {
            println!("cargo:warning=Shader {} not found, skipping", src.display());
            continue;
        }

        // Compile .metal → .air (Metal intermediate representation)
        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx",
                "metal", "-c",
                src.to_str().unwrap(),
                "-o", &air,
            ])
            .status()
            .expect("xcrun metal failed to launch — is Xcode installed?");

        if !status.success() {
            panic!("metal compilation failed for {shader}.metal");
        }

        air_files.push(air);
    }

    if air_files.is_empty() {
        println!("cargo:warning=No shaders compiled; MetalKernels.metallib will not be built");
        return;
    }

    // Link all .air files into a single MetalKernels.metallib
    let combined = format!("{out}/MetalKernels.metallib");
    let mut metallib_cmd = Command::new("xcrun");
    metallib_cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        metallib_cmd.arg(air);
    }
    metallib_cmd.args(["-o", &combined]);

    let status = metallib_cmd
        .status()
        .expect("xcrun metallib failed to launch");
    if !status.success() {
        panic!("metallib link step failed");
    }

    // Copy MetalKernels.metallib to the workspace root so that
    // device.new_library_with_file("MetalKernels.metallib") resolves when
    // the process is run with the workspace root as its working directory
    // (which is the default for `cargo run`).
    let dest = workspace_root.join("MetalKernels.metallib");
    std::fs::copy(&combined, &dest)
        .unwrap_or_else(|e| panic!("failed to copy metallib to workspace root: {e}"));

    println!("cargo:rerun-if-changed=build.rs");
}
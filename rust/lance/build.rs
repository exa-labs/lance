use std::env;
use std::fs;
use std::path::PathBuf;

// Generate a constant slice containing all features enabled for this crate.
// Cargo exposes enabled features to build scripts via CARGO_FEATURE_* env vars.
fn main() {
    let mut features: Vec<String> = env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .map(|name| name.to_lowercase().replace('_', "-"))
        .collect();

    features.sort();
    features.dedup();

    let contents = format!("pub const BUILD_FEATURES: &[&str] = &{:?};\n", features);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let out_path = out_dir.join("features.rs");
    fs::write(out_path, contents).expect("write features.rs");
}

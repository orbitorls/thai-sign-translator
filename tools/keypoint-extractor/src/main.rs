fn main() {
    if let Err(err) = keypoint_extractor::run_cli() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

use std::collections::HashSet;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

#[derive(Debug, Clone)]
pub struct ExtractionArgs {
    pub input_csv: PathBuf,
    pub out_dir: PathBuf,
    pub source: String,
    pub python: String,
    pub limit: Option<usize>,
    pub fps: i32,
    pub seed: u64,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct RunOutput {
    pub stdout: String,
    pub stderr: String,
    pub success: bool,
    pub summary: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct SegmentRow {
    segment_id: String,
    video_path: String,
    text: String,
    start_ms: f64,
    end_ms: f64,
    split: String,
}

pub fn run_cli() -> Result<(), String> {
    let args = parse_args(env::args_os().skip(1))?;
    let output = run_job(&args)?;

    if !output.stdout.is_empty() {
        print!("{}", output.stdout);
    }
    if !output.stderr.is_empty() {
        eprint!("{}", output.stderr);
    }
    if output.success {
        Ok(())
    } else {
        Err(output.summary)
    }
}

pub fn run_job(args: &ExtractionArgs) -> Result<RunOutput, String> {
    validate_output_dir(&args.out_dir)?;
    let rows = load_rows(&args.input_csv)?;
    if rows.is_empty() {
        return Err("input CSV has no usable rows".to_string());
    }

    if args.dry_run {
        let summary = format!(
            "validated {} rows from {} for output {}",
            rows.len(),
            args.input_csv.display(),
            args.out_dir.display()
        );
        return Ok(RunOutput {
            stdout: format!("{summary}\n"),
            stderr: String::new(),
            success: true,
            summary,
        });
    }

    let script = python_helper_path()?;
    if !script.is_file() {
        return Err(format!("Python helper not found: {}", script.display()));
    }

    let output = run_python(args, &script)?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let success = output.status.success();
    let summary = if success {
        "Python extractor completed successfully".to_string()
    } else {
        format!("Python extractor failed with status {}", output.status)
    };

    Ok(RunOutput {
        stdout,
        stderr,
        success,
        summary,
    })
}

fn parse_args<I>(mut iter: I) -> Result<ExtractionArgs, String>
where
    I: Iterator<Item = OsString>,
{
    let mut input_csv: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut source = String::from("custom");
    let mut python = String::from("python");
    let mut limit: Option<usize> = None;
    let mut fps = 0;
    let mut seed = 42_u64;
    let mut dry_run = false;

    while let Some(flag) = iter.next() {
        let flag = flag.to_string_lossy().into_owned();
        match flag.as_str() {
            "--input-csv" => input_csv = Some(PathBuf::from(next_value(&mut iter, "--input-csv")?)),
            "--out-dir" => out_dir = Some(PathBuf::from(next_value(&mut iter, "--out-dir")?)),
            "--source" => source = next_value(&mut iter, "--source")?,
            "--python" => python = next_value(&mut iter, "--python")?,
            "--limit" => {
                let value = next_value(&mut iter, "--limit")?;
                limit = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --limit value: {value}"))?,
                );
            }
            "--fps" => {
                let value = next_value(&mut iter, "--fps")?;
                fps = value
                    .parse::<i32>()
                    .map_err(|_| format!("invalid --fps value: {value}"))?;
            }
            "--seed" => {
                let value = next_value(&mut iter, "--seed")?;
                seed = value
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --seed value: {value}"))?;
            }
            "--dry-run" => dry_run = true,
            "--help" | "-h" => {
                print_usage();
                return Err(String::from("help requested"));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let input_csv = input_csv.ok_or_else(|| "missing --input-csv".to_string())?;
    let out_dir = out_dir.ok_or_else(|| "missing --out-dir".to_string())?;
    Ok(ExtractionArgs {
        input_csv,
        out_dir,
        source,
        python,
        limit,
        fps,
        seed,
        dry_run,
    })
}

fn print_usage() {
    eprintln!(
        "Usage: keypoint-extractor --input-csv PATH --out-dir PATH [--source NAME] [--python PYTHON] [--limit N] [--fps N] [--seed N] [--dry-run]"
    );
}

fn next_value<I>(iter: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = OsString>,
{
    iter.next()
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| format!("missing value for {flag}"))
}

pub fn validate_output_dir(path: &Path) -> Result<(), String> {
    if path.exists() && !path.is_dir() {
        return Err(format!(
            "output path is not a directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn python_helper_path() -> Result<PathBuf, String> {
    repo_root().map(|root| root.join("scripts").join("extract_dataset_keypoints.py"))
}

fn repo_root() -> Result<PathBuf, String> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(|path| path.parent())
        .map(Path::to_path_buf)
        .ok_or_else(|| "could not resolve repo root from crate dir".to_string())
}

fn load_rows(path: &Path) -> Result<Vec<SegmentRow>, String> {
    let data = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut lines = data.lines().filter(|line| !line.trim().is_empty());
    let header_line = lines
        .next()
        .ok_or_else(|| format!("input CSV is empty: {}", path.display()))?;
    let headers = parse_csv_line(header_line);
    let required = [
        "segment_id",
        "video_path",
        "text",
        "start_ms",
        "end_ms",
        "split",
    ];
    let header_set: HashSet<_> = headers.iter().map(|value| value.as_str()).collect();
    for required_name in required {
        if !header_set.contains(required_name) {
            return Err(format!(
                "missing required column {required_name:?} in {}",
                path.display()
            ));
        }
    }

    let mut rows = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut row_index = 1;
    for line in lines {
        row_index += 1;
        let values = parse_csv_line(line);
        if values.len() != headers.len() {
            return Err(format!(
                "row {row_index} has {} fields but header has {}",
                values.len(),
                headers.len()
            ));
        }
        let get = |name: &str| -> Result<String, String> {
            let idx = headers
                .iter()
                .position(|header| header == name)
                .ok_or_else(|| format!("missing column {name:?}"))?;
            Ok(values[idx].clone())
        };
        let segment_id = get("segment_id")?.trim().to_string();
        if is_blankish(&segment_id) {
            return Err(format!("row {row_index} has empty segment_id"));
        }
        if !seen_ids.insert(segment_id.clone()) {
            return Err(format!(
                "duplicate segment_id at row {row_index}: {segment_id}"
            ));
        }
        let video_path = get("video_path")?.trim().to_string();
        if is_blankish(&video_path) {
            return Err(format!("row {row_index} has empty video_path"));
        }
        let resolved_video_path = resolve_path(path, &video_path)?;
        if !resolved_video_path.is_file() {
            return Err(format!(
                "row {row_index} video file not found: {}",
                resolved_video_path.display()
            ));
        }
        let text = get("text")?.trim().to_string();
        if is_blankish(&text) {
            return Err(format!("row {row_index} has empty text"));
        }
        let start_ms = get("start_ms")?
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("row {row_index} has invalid start_ms"))?;
        let end_ms = get("end_ms")?
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("row {row_index} has invalid end_ms"))?;
        if start_ms >= end_ms {
            return Err(format!("row {row_index} has start_ms >= end_ms"));
        }
        let split = get("split")?.trim().to_string();
        if is_blankish(&split) || !matches!(split.as_str(), "train" | "val" | "test") {
            return Err(format!("row {row_index} has invalid split {split:?}"));
        }
        rows.push(SegmentRow {
            segment_id,
            video_path: resolved_video_path.to_string_lossy().into_owned(),
            text,
            start_ms,
            end_ms,
            split,
        });
    }

    Ok(rows)
}

fn resolve_path(csv_path: &Path, raw_path: &str) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        return Ok(candidate);
    }
    let base_dir = csv_path
        .parent()
        .ok_or_else(|| format!("cannot resolve parent directory for {}", csv_path.display()))?;
    Ok(base_dir.join(candidate))
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && matches!(chars.peek(), Some('"')) {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            ',' if !in_quotes => {
                values.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    values.push(current.trim().to_string());
    values
}

fn is_blankish(value: &str) -> bool {
    let lowered = value.trim().to_ascii_lowercase();
    lowered.is_empty() || lowered == "nan" || lowered == "none"
}

fn run_python(args: &ExtractionArgs, script: &Path) -> Result<OutputCapture, String> {
    let mut cmd = Command::new(&args.python);
    cmd.arg(script)
        .arg("--input-csv")
        .arg(&args.input_csv)
        .arg("--out-dir")
        .arg(&args.out_dir)
        .arg("--source")
        .arg(&args.source)
        .arg("--fps")
        .arg(args.fps.to_string())
        .arg("--seed")
        .arg(args.seed.to_string());
    if let Some(limit) = args.limit {
        cmd.arg("--limit").arg(limit.to_string());
    }
    if args.dry_run {
        cmd.arg("--dry-run");
    }
    let output = cmd
        .output()
        .map_err(|err| format!("failed to start python extractor: {err}"))?;
    Ok(OutputCapture {
        stdout: output.stdout,
        stderr: output.stderr,
        status: output.status,
    })
}

struct OutputCapture {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: ExitStatus,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file_name(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        env::temp_dir().join(format!("{prefix}_{stamp}.csv"))
    }

    fn sample_args(csv_path: PathBuf, out_dir: PathBuf, dry_run: bool) -> ExtractionArgs {
        ExtractionArgs {
            input_csv: csv_path,
            out_dir,
            source: "custom".to_string(),
            python: "python".to_string(),
            limit: None,
            fps: 0,
            seed: 42,
            dry_run,
        }
    }

    #[test]
    fn load_rows_accepts_simple_segment_csv() {
        let path = temp_file_name("keypoint_rows_ok");
        let video_path = path.parent().expect("parent").join("video.mp4");
        fs::write(&video_path, b"fake").expect("write video");
        fs::write(
            &path,
            "segment_id,video_path,text,start_ms,end_ms,split\ns1,video.mp4,hello,0,1000,train\n",
        )
        .expect("write csv");

        let rows = load_rows(&path).expect("rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].segment_id, "s1");
        assert_eq!(rows[0].split, "train");
        assert!(rows[0].end_ms > rows[0].start_ms);

        let _ = fs::remove_file(path);
        let _ = fs::remove_file(video_path);
    }

    #[test]
    fn load_rows_rejects_duplicate_segment_ids() {
        let path = temp_file_name("keypoint_rows_dupe");
        let video_path = path.parent().expect("parent").join("video.mp4");
        fs::write(&video_path, b"fake").expect("write video");
        fs::write(
            &path,
            "segment_id,video_path,text,start_ms,end_ms,split\ns1,video.mp4,hello,0,1000,train\ns1,video.mp4,bye,1000,2000,val\n",
        )
        .expect("write csv");

        let err = load_rows(&path).unwrap_err();
        assert!(err.contains("duplicate segment_id"));

        let _ = fs::remove_file(path);
        let _ = fs::remove_file(video_path);
    }

    #[test]
    fn parse_csv_line_handles_quoted_commas() {
        let values = parse_csv_line("s1,\"video,01.mp4\",hello,0,1000,train");
        assert_eq!(values[1], "video,01.mp4");
    }

    #[test]
    fn dry_run_reports_validation_summary_without_creating_output_dir() {
        let csv_path = temp_file_name("keypoint_dry_run");
        let video_path = csv_path.parent().expect("parent").join("video.mp4");
        fs::write(&video_path, b"fake").expect("write video");
        fs::write(
            &csv_path,
            "segment_id,video_path,text,start_ms,end_ms,split\ns1,video.mp4,hello,0,1000,train\n",
        )
        .expect("write csv");
        let out_dir = csv_path.parent().expect("parent").join("out");
        let args = sample_args(csv_path.clone(), out_dir.clone(), true);

        let result = run_job(&args).expect("run");

        assert!(result.success);
        assert!(result.stdout.contains("validated 1 rows"));
        assert!(!out_dir.exists());

        let _ = fs::remove_file(csv_path);
        let _ = fs::remove_file(video_path);
    }
}

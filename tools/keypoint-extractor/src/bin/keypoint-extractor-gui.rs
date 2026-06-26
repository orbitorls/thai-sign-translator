use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::thread;

use native_windows_gui as nwg;

use keypoint_extractor::{run_job, ExtractionArgs, RunOutput};

#[derive(Clone, Copy, Debug)]
enum JobKind {
    DryRun,
    Extract,
}

struct WorkerMessage {
    job: JobKind,
    result: Result<RunOutput, String>,
}

#[derive(Default)]
struct KeypointExtractorApp {
    window: nwg::Window,
    layout: nwg::GridLayout,
    status_label: nwg::Label,
    input_csv_label: nwg::Label,
    input_csv_input: nwg::TextInput,
    input_csv_browse: nwg::Button,
    out_dir_label: nwg::Label,
    out_dir_input: nwg::TextInput,
    out_dir_browse: nwg::Button,
    python_label: nwg::Label,
    python_input: nwg::TextInput,
    python_browse: nwg::Button,
    source_label: nwg::Label,
    source_input: nwg::TextInput,
    fps_label: nwg::Label,
    fps_input: nwg::TextInput,
    seed_label: nwg::Label,
    seed_input: nwg::TextInput,
    limit_label: nwg::Label,
    limit_input: nwg::TextInput,
    dry_run_button: nwg::Button,
    extract_button: nwg::Button,
    clear_button: nwg::Button,
    logs_label: nwg::Label,
    log_box: nwg::TextBox,
    csv_dialog: nwg::FileDialog,
    dir_dialog: nwg::FileDialog,
    python_dialog: nwg::FileDialog,
    notice: nwg::Notice,
    event_handler: RefCell<Option<nwg::EventHandler>>,
    worker_result: Arc<Mutex<Option<WorkerMessage>>>,
    running: Cell<bool>,
    active_job: Cell<Option<JobKind>>,
}

impl KeypointExtractorApp {
    fn build() -> Result<Rc<Self>, nwg::NwgError> {
        nwg::init()?;
        let _ = nwg::Font::set_global_family("Segoe UI");
        let default_dir = std::env::current_dir()
            .ok()
            .map(|path| path.display().to_string())
            .unwrap_or_default();

        let mut app = KeypointExtractorApp::default();

        nwg::Window::builder()
            .flags(nwg::WindowFlags::WINDOW | nwg::WindowFlags::VISIBLE)
            .size((1080, 820))
            .position((200, 120))
            .title("Keypoint Extractor")
            .build(&mut app.window)?;

        nwg::Label::builder()
            .text("Ready")
            .parent(&app.window)
            .build(&mut app.status_label)?;

        nwg::Label::builder()
            .text("Input CSV")
            .parent(&app.window)
            .build(&mut app.input_csv_label)?;
        nwg::TextInput::builder()
            .text("")
            .focus(true)
            .parent(&app.window)
            .build(&mut app.input_csv_input)?;
        nwg::Button::builder()
            .text("Browse")
            .parent(&app.window)
            .build(&mut app.input_csv_browse)?;

        nwg::Label::builder()
            .text("Output folder")
            .parent(&app.window)
            .build(&mut app.out_dir_label)?;
        nwg::TextInput::builder()
            .text(&default_dir)
            .parent(&app.window)
            .build(&mut app.out_dir_input)?;
        nwg::Button::builder()
            .text("Browse")
            .parent(&app.window)
            .build(&mut app.out_dir_browse)?;

        nwg::Label::builder()
            .text("Python")
            .parent(&app.window)
            .build(&mut app.python_label)?;
        nwg::TextInput::builder()
            .text("python")
            .parent(&app.window)
            .build(&mut app.python_input)?;
        nwg::Button::builder()
            .text("Browse")
            .parent(&app.window)
            .build(&mut app.python_browse)?;

        nwg::Label::builder()
            .text("Source")
            .parent(&app.window)
            .build(&mut app.source_label)?;
        nwg::TextInput::builder()
            .text("custom")
            .parent(&app.window)
            .build(&mut app.source_input)?;

        nwg::Label::builder()
            .text("FPS")
            .parent(&app.window)
            .build(&mut app.fps_label)?;
        nwg::TextInput::builder()
            .text("0")
            .parent(&app.window)
            .build(&mut app.fps_input)?;
        nwg::Label::builder()
            .text("Seed")
            .parent(&app.window)
            .build(&mut app.seed_label)?;
        nwg::TextInput::builder()
            .text("42")
            .parent(&app.window)
            .build(&mut app.seed_input)?;

        nwg::Label::builder()
            .text("Limit")
            .parent(&app.window)
            .build(&mut app.limit_label)?;
        nwg::TextInput::builder()
            .text("")
            .parent(&app.window)
            .build(&mut app.limit_input)?;

        nwg::Button::builder()
            .text("Dry Run")
            .parent(&app.window)
            .build(&mut app.dry_run_button)?;
        nwg::Button::builder()
            .text("Extract")
            .parent(&app.window)
            .build(&mut app.extract_button)?;
        nwg::Button::builder()
            .text("Clear Log")
            .parent(&app.window)
            .build(&mut app.clear_button)?;

        nwg::Label::builder()
            .text("Log")
            .parent(&app.window)
            .build(&mut app.logs_label)?;
        nwg::TextBox::builder()
            .text("")
            .readonly(true)
            .parent(&app.window)
            .build(&mut app.log_box)?;

        nwg::FileDialog::builder()
            .action(nwg::FileDialogAction::Open)
            .title("Select input CSV")
            .build(&mut app.csv_dialog)?;
        nwg::FileDialog::builder()
            .action(nwg::FileDialogAction::OpenDirectory)
            .title("Select output folder")
            .build(&mut app.dir_dialog)?;
        nwg::FileDialog::builder()
            .action(nwg::FileDialogAction::Open)
            .title("Select Python executable")
            .build(&mut app.python_dialog)?;
        nwg::Notice::builder()
            .parent(&app.window)
            .build(&mut app.notice)?;

        nwg::GridLayout::builder()
            .parent(&app.window)
            .spacing(6)
            .margin([10, 10, 10, 10])
            .child_item(nwg::GridLayoutItem::new(&app.status_label, 0, 0, 4, 1))
            .child_item(nwg::GridLayoutItem::new(&app.input_csv_label, 0, 1, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.input_csv_input, 1, 1, 2, 1))
            .child_item(nwg::GridLayoutItem::new(&app.input_csv_browse, 3, 1, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.out_dir_label, 0, 2, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.out_dir_input, 1, 2, 2, 1))
            .child_item(nwg::GridLayoutItem::new(&app.out_dir_browse, 3, 2, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.python_label, 0, 3, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.python_input, 1, 3, 2, 1))
            .child_item(nwg::GridLayoutItem::new(&app.python_browse, 3, 3, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.source_label, 0, 4, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.source_input, 1, 4, 3, 1))
            .child_item(nwg::GridLayoutItem::new(&app.fps_label, 0, 5, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.fps_input, 1, 5, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.seed_label, 2, 5, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.seed_input, 3, 5, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.limit_label, 0, 6, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.limit_input, 1, 6, 3, 1))
            .child_item(nwg::GridLayoutItem::new(&app.dry_run_button, 0, 7, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.extract_button, 1, 7, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.clear_button, 2, 7, 1, 1))
            .child_item(nwg::GridLayoutItem::new(&app.logs_label, 0, 8, 4, 1))
            .child_item(nwg::GridLayoutItem::new(&app.log_box, 0, 9, 4, 1))
            .build(&app.layout)?;

        app.log_box.set_text(
            "Choose an input CSV and output folder, then click Dry Run or Extract.\r\n",
        );

        let app = Rc::new(app);
        let weak_app: Weak<KeypointExtractorApp> = Rc::downgrade(&app);
        let handler = nwg::full_bind_event_handler(&app.window.handle, move |evt, _evt_data, handle| {
            if let Some(app) = weak_app.upgrade() {
                app.process_event(evt, handle);
            }
        });
        *app.event_handler.borrow_mut() = Some(handler);

        Ok(app)
    }

    fn process_event(&self, evt: nwg::Event, handle: nwg::ControlHandle) {
        use nwg::Event as E;

        match evt {
            E::OnButtonClick if &handle == &self.input_csv_browse => self.pick_csv(),
            E::OnButtonClick if &handle == &self.out_dir_browse => self.pick_output_dir(),
            E::OnButtonClick if &handle == &self.python_browse => self.pick_python(),
            E::OnButtonClick if &handle == &self.dry_run_button => self.start_job(true),
            E::OnButtonClick if &handle == &self.extract_button => self.start_job(false),
            E::OnButtonClick if &handle == &self.clear_button => self.clear_log(),
            E::OnNotice if &handle == &self.notice => self.finish_job(),
            E::OnWindowClose if &handle == &self.window => nwg::stop_thread_dispatch(),
            _ => {}
        }
    }

    fn pick_csv(&self) {
        if self.csv_dialog.run(Some(&self.window)) {
            if let Ok(path) = self.csv_dialog.get_selected_item() {
                self.input_csv_input.set_text(&path.to_string_lossy());
            }
        }
    }

    fn pick_output_dir(&self) {
        if self.dir_dialog.run(Some(&self.window)) {
            if let Ok(path) = self.dir_dialog.get_selected_item() {
                self.out_dir_input.set_text(&path.to_string_lossy());
            }
        }
    }

    fn pick_python(&self) {
        if self.python_dialog.run(Some(&self.window)) {
            if let Ok(path) = self.python_dialog.get_selected_item() {
                self.python_input.set_text(&path.to_string_lossy());
            }
        }
    }

    fn set_running(&self, running: bool, status: &str) {
        self.running.set(running);
        if !running {
            self.active_job.set(None);
        }
        self.status_label.set_text(status);
        let enabled = !running;
        self.input_csv_browse.set_enabled(enabled);
        self.out_dir_browse.set_enabled(enabled);
        self.python_browse.set_enabled(enabled);
        self.dry_run_button.set_enabled(enabled);
        self.extract_button.set_enabled(enabled);
        self.clear_button.set_enabled(enabled);
    }

    fn append_log(&self, line: &str) {
        let mut current = self.log_box.text();
        if !current.is_empty() && !current.ends_with("\r\n") {
            current.push_str("\r\n");
        }
        current.push_str(line);
        current.push_str("\r\n");
        self.log_box.set_text(&current);
    }

    fn clear_log(&self) {
        self.log_box.set_text("");
    }

    fn parse_args(&self, dry_run: bool) -> Result<ExtractionArgs, String> {
        let input_csv = PathBuf::from(self.input_csv_input.text().trim());
        if input_csv.as_os_str().is_empty() {
            return Err("input CSV is required".to_string());
        }

        let out_dir = PathBuf::from(self.out_dir_input.text().trim());
        if out_dir.as_os_str().is_empty() {
            return Err("output folder is required".to_string());
        }

        let source = self.source_input.text().trim().to_string();
        let python = self.python_input.text().trim().to_string();
        let fps = self
            .fps_input
            .text()
            .trim()
            .parse::<i32>()
            .map_err(|_| "FPS must be an integer".to_string())?;
        let seed = self
            .seed_input
            .text()
            .trim()
            .parse::<u64>()
            .map_err(|_| "Seed must be an integer".to_string())?;
        let limit = {
            let value = self.limit_input.text();
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(
                    trimmed
                        .parse::<usize>()
                        .map_err(|_| "Limit must be an integer".to_string())?,
                )
            }
        };

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

    fn start_job(&self, dry_run: bool) {
        if self.running.get() {
            return;
        }

        let args = match self.parse_args(dry_run) {
            Ok(args) => args,
            Err(err) => {
                self.status_label.set_text(&err);
                self.append_log(&format!("[error] {err}"));
                return;
            }
        };

        let job = if dry_run {
            JobKind::DryRun
        } else {
            JobKind::Extract
        };
        self.active_job.set(Some(job));
        self.set_running(true, match job {
            JobKind::DryRun => "Validating CSV...",
            JobKind::Extract => "Running extractor...",
        });
        self.append_log(match job {
            JobKind::DryRun => "== DRY RUN ==",
            JobKind::Extract => "== EXTRACT ==",
        });

        let result_slot = Arc::clone(&self.worker_result);
        let notice = self.notice.sender();
        thread::spawn(move || {
            let result = run_job(&args);
            if let Ok(mut slot) = result_slot.lock() {
                *slot = Some(WorkerMessage { job, result });
            }
            notice.notice();
        });
    }

    fn finish_job(&self) {
        let message = self
            .worker_result
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());

        let Some(message) = message else {
            self.set_running(false, "Ready");
            return;
        };

        self.set_running(false, "Ready");
        self.active_job.set(None);

        match message.result {
            Ok(output) => {
                if output.success {
                    self.status_label.set_text(match message.job {
                        JobKind::DryRun => "Validation finished",
                        JobKind::Extract => "Extraction finished",
                    });
                } else {
                    self.status_label
                        .set_text("Extractor returned a non-zero exit status");
                }
                self.append_log(&format!("[summary] {}", output.summary));
                if !output.stdout.trim().is_empty() {
                    self.append_log("stdout:");
                    self.append_log(output.stdout.trim_end());
                }
                if !output.stderr.trim().is_empty() {
                    self.append_log("stderr:");
                    self.append_log(output.stderr.trim_end());
                }
            }
            Err(err) => {
                self.status_label.set_text(&err);
                self.append_log(&format!("[error] {err}"));
            }
        }
    }
}

impl Drop for KeypointExtractorApp {
    fn drop(&mut self) {
        if let Some(handler) = self.event_handler.borrow_mut().take() {
            nwg::unbind_event_handler(&handler);
        }
    }
}

fn main() {
    match KeypointExtractorApp::build() {
        Ok(_app) => nwg::dispatch_thread_events(),
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

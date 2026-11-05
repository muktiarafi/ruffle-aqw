use std::ptr::NonNull;

use windows_sys::Win32::Storage::FileSystem::{FILE_TYPE_DISK, FILE_TYPE_PIPE, GetFileType};
use windows_sys::Win32::System::Console::{
    ATTACH_PARENT_PROCESS, AttachConsole, FreeConsole, GetStdHandle, STD_OUTPUT_HANDLE,
};

/// RAII guard for the attached parent console.
/// Frees the console on drop if it was successfully attached.
pub(super) struct Console {
    attached: bool,
}

impl Console {
    // When linked with the windows subsystem windows won't automatically attach
    // to the console of the parent process, so we do it explicitly. This fails
    // silently if the parent has no console.
    //
    // However, if stdout/stderr are already redirected (e.g., `ruffle.exe > file.txt`),
    // we should NOT attach to the console as that would bypass the redirection.
    // See: https://github.com/ruffle-rs/ruffle/issues/9145
    pub(super) fn attach() -> Self {
        // Check if stdout is already redirected to a file or pipe
        // SAFETY: STD_OUTPUT_HANDLE is a valid standard device constant.
        let stdout_handle = NonNull::new(unsafe { GetStdHandle(STD_OUTPUT_HANDLE) });

        let should_attach = stdout_handle.is_none_or(|mut handle| {
            // SAFETY: GetFileType accepts any handle value including INVALID_HANDLE_VALUE,
            // returning FILE_TYPE_UNKNOWN in that case.
            let file_type = unsafe { GetFileType(handle.as_mut()) };

            // If output is redirected to a file or pipe, don't attach to console
            // as that would bypass the redirection
            !matches!(file_type, FILE_TYPE_DISK | FILE_TYPE_PIPE)
        });

        let attached = if should_attach {
            // Otherwise, attach to parent console for interactive use
            // SAFETY: ATTACH_PARENT_PROCESS is a valid constant for AttachConsole.
            // This call fails silently if the parent has no console.
            (unsafe { AttachConsole(ATTACH_PARENT_PROCESS) }) != 0
        } else {
            false
        };

        Self { attached }
    }
}

impl Drop for Console {
    fn drop(&mut self) {
        if self.attached {
            // Without explicitly detaching, cmd won't redraw its prompt.
            // SAFETY: We only call FreeConsole if it was previously successfully attached.
            unsafe { FreeConsole() };
        }
    }
}

/// AQW build (Mugetsu): runs at high priority and opts out of Windows 11's "EcoQoS" power
/// throttling, which otherwise lowers the clock speed of a process Windows considers in the
/// background (e.g. while another window has focus) and caused frame drops. The old .NET
/// launcher did this from outside; doing it here keeps the launcher trivial.
pub(super) fn aqw_tune_process() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, HIGH_PRIORITY_CLASS, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetPriorityClass, SetProcessInformation,
    };
    // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no closing; the struct
    // pointer and size describe a valid PROCESS_POWER_THROTTLING_STATE. Failures are ignored.
    unsafe {
        let process = GetCurrentProcess();
        SetPriorityClass(process, HIGH_PRIORITY_CLASS);
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        SetProcessInformation(
            process,
            ProcessPowerThrottling,
            (&raw const state).cast(),
            size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        );
    }
}

/// AQW build (Mugetsu): a crash in native code (almost always a graphics driver, e.g. Intel's
/// Vulkan driver igvk64.dll) used to make Mugetsu vanish without a word: Rust's panic hook only
/// sees Rust panics. This catches it, writes where it happened to the log and says what
/// happened instead. If the Vulkan driver crashed while Mugetsu was starting, it remembers that
/// (see `mugetsu::save_vulkan_crash`) and offers to start Mugetsu again, which then uses
/// DirectX 12.
pub(super) fn install_crash_handler() {
    use windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter;
    // SAFETY: `crash_filter` has the signature Windows expects and stays valid for the whole
    // process lifetime.
    unsafe { SetUnhandledExceptionFilter(Some(crash_filter)) };
}

unsafe extern "system" fn crash_filter(
    info: *const windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    const EXCEPTION_EXECUTE_HANDLER: i32 = 1;
    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_NONCONTINUABLE: u32 = 1;
    const EXCEPTION_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;

    // Software exceptions whose code isn't a warning or an error (the top bit clear: "success"
    // or "informational" severity) aren't crashes, but notifications that the code raising them
    // expects to be ignored without a debugger. DirectX raises 0x87A and 0x87D in KERNELBASE.dll
    // this way on some PCs (e.g. while Mugetsu closes), which used to show a "crashed" message.
    // Carry on after the RaiseException call instead.
    // SAFETY: Windows passes a valid EXCEPTION_POINTERS with a valid exception record.
    let (code, flags) = unsafe {
        let record = &*(*info).ExceptionRecord;
        (record.ExceptionCode as u32, record.ExceptionFlags)
    };
    if code & 0x8000_0000 == 0 && flags & EXCEPTION_NONCONTINUABLE == 0 {
        static IGNORED: AtomicU32 = AtomicU32::new(0);
        if IGNORED.fetch_add(1, Ordering::Relaxed) < 10 {
            tracing::warn!("Ignored a non-fatal exception 0x{code:08X}");
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    static HANDLING: AtomicBool = AtomicBool::new(false);
    if HANDLING.swap(true, Ordering::SeqCst) {
        // Another thread crashed too (or this one crashed again inside the handler): leave the
        // first report on screen; the process ends when it is closed.
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }

    // SAFETY: as above.
    let address = unsafe { (*(*info).ExceptionRecord).ExceptionAddress as usize };
    let module = module_at(address);
    let in_mugetsu = module.as_ref().is_none_or(|module| module.is_main_exe);
    let file_name = module.as_ref().map_or_else(
        || "an unknown module".to_string(),
        |module| {
            module.path.file_name().map_or_else(
                || module.path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            )
        },
    );
    let vulkan_starting = crate::mugetsu::vulkan_starting();
    tracing::error!(
        "fatal native crash: exception 0x{code:08X} in {} at offset 0x{:X} (Vulkan starting: \
        {vulkan_starting})\n{}",
        module
            .as_ref()
            .map_or_else(|| "an unknown module".to_string(), |m| m.path.display().to_string()),
        module.as_ref().map_or(0, |m| m.offset),
        std::backtrace::Backtrace::force_capture()
    );

    if code == EXCEPTION_ILLEGAL_INSTRUCTION && in_mugetsu {
        message_box(
            "Mugetsu can't run on this PC: it needs a processor with AVX2 (Intel Core 4th \
            generation from 2013 or newer, AMD Ryzen or newer). Some budget Celeron and Pentium \
            laptop processors don't have it.",
            false,
        );
    } else if vulkan_starting
        && let Some(module) = module.as_ref().filter(|module| module.is_graphics_driver())
    {
        crate::mugetsu::save_vulkan_crash(&module.path);
        let restart = message_box(
            &format!(
                "The graphics driver crashed while Mugetsu was starting Vulkan (in {file_name}).\n\n\
                Mugetsu will use DirectX 12 on this PC instead, which runs about as well. On \
                a laptop's built-in graphics, set Quality to Low in AQW's options if the game \
                runs slowly. Mugetsu tries Vulkan again by itself after you update your \
                graphics driver.\n\n\
                Click OK to start Mugetsu again."
            ),
            true,
        );
        if restart {
            start_again();
        }
    } else {
        let hint = if in_mugetsu {
            String::new()
        } else {
            format!(
                "It happened inside {file_name}, which isn't part of Mugetsu: usually the \
                graphics driver, or a program that hooks into games (overlays, recorders). \
                Updating your graphics driver often fixes it.\n\n"
            )
        };
        message_box(
            &format!(
                "Mugetsu crashed (error 0x{code:08X} in {file_name}) and has to close.\n\n{hint}\
                Details were saved to %LOCALAPPDATA%\\ruffle\\log\\ruffle.log (it is replaced \
                the next time Mugetsu starts)."
            ),
            false,
        );
    }
    // End right here: a normal exit runs every DLL's shutdown code, and a crashed graphics
    // driver can hang in its own.
    // SAFETY: terminating the current process (pseudo-handle, never closed).
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
        TerminateProcess(GetCurrentProcess(), code);
    }
    EXCEPTION_EXECUTE_HANDLER
}

/// AQW build: `RUFFLE_TEST_NATIVE_CRASH=driver` crashes inside a Windows DLL (counts as a
/// graphics driver crash when `RUFFLE_TEST_DRIVER_FILE` names that DLL, e.g. ntdll.dll),
/// `=own` inside Mugetsu itself, to test `install_crash_handler`.
pub(super) fn test_native_crash() {
    use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
    match std::env::var("RUFFLE_TEST_NATIVE_CRASH").as_deref() {
        // SAFETY: deliberately invalid: makes kernelbase.dll write to address 8.
        Ok("driver") => unsafe {
            GetModuleFileNameW(std::ptr::null_mut(), 8 as *mut u16, 260);
        },
        // SAFETY: deliberately invalid: writes to address 8.
        Ok("own") => unsafe { std::ptr::write_volatile(8 as *mut u32, 0) },
        _ => {}
    }
}

struct CrashModule {
    path: std::path::PathBuf,
    offset: usize,
    is_main_exe: bool,
}

impl CrashModule {
    /// Graphics drivers are installed in Windows' driver store (a new folder per driver
    /// version); older ones put their Vulkan driver (Intel's igvk64.dll, NVIDIA's nvoglv64.dll,
    /// AMD's amdvlk64.dll) in System32. Only a crash in one of those switches Mugetsu to
    /// DirectX 12; one in an overlay, antivirus or other Windows file doesn't.
    fn is_graphics_driver(&self) -> bool {
        let path = self.path.to_string_lossy().to_ascii_lowercase();
        let file = path.rsplit('\\').next().unwrap_or_default();
        path.contains(r"\driverstore\filerepository\")
            || ["igvk64.dll", "nvoglv64.dll", "amdvlk64.dll"].contains(&file)
            || std::env::var("RUFFLE_TEST_DRIVER_FILE")
                .is_ok_and(|test| test.eq_ignore_ascii_case(file))
    }
}

/// The loaded module (exe or DLL) that contains `address`.
fn module_at(address: usize) -> Option<CrashModule> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleFileNameW, GetModuleHandleExW, GetModuleHandleW,
    };

    let mut module: HMODULE = std::ptr::null_mut();
    let mut buffer = [0u16; 1024];
    // SAFETY: with FROM_ADDRESS the "name" is an address inside the module; UNCHANGED_REFCOUNT
    // means nothing must be released. The buffer pointer and length match.
    let length = unsafe {
        if GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address as *const u16,
            &mut module,
        ) == 0
        {
            return None;
        }
        GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) as usize
    };
    // SAFETY: a null name returns the handle of the process's exe.
    let main_exe = unsafe { GetModuleHandleW(std::ptr::null()) };
    Some(CrashModule {
        path: std::ffi::OsString::from_wide(&buffer[..length.min(buffer.len())]).into(),
        offset: address.wrapping_sub(module as usize),
        is_main_exe: module == main_exe,
    })
}

/// Shows an error message box; with `ok_cancel`, returns whether OK was clicked.
fn message_box(text: &str, ok_cancel: bool) -> bool {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        IDOK, MB_ICONERROR, MB_OK, MB_OKCANCEL, MB_SETFOREGROUND, MB_TOPMOST, MessageBoxW,
    };
    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let (text, title) = (wide(text), wide("Mugetsu"));
    let buttons = if ok_cancel { MB_OKCANCEL } else { MB_OK };
    // SAFETY: both strings are NUL-terminated UTF-16 that outlive the call.
    let result = unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            text.as_ptr(),
            title.as_ptr(),
            buttons | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST,
        )
    };
    result == IDOK
}

/// Starts Mugetsu again with the same command line.
pub fn start_again() {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Environment::GetCommandLineW;
    use windows_sys::Win32::System::Threading::{
        CreateProcessW, PROCESS_INFORMATION, STARTUPINFOW,
    };
    // SAFETY: GetCommandLineW returns this process's NUL-terminated command line, copied (with
    // its terminator) because CreateProcessW may modify the buffer. The structs are plain data
    // for which all-zero is valid, and the returned handles are closed.
    unsafe {
        let command_line = GetCommandLineW();
        let mut length = 0;
        while *command_line.add(length) != 0 {
            length += 1;
        }
        let mut buffer = std::slice::from_raw_parts(command_line, length + 1).to_vec();
        let mut startup: STARTUPINFOW = std::mem::zeroed();
        startup.cb = size_of::<STARTUPINFOW>() as u32;
        let mut process: PROCESS_INFORMATION = std::mem::zeroed();
        if CreateProcessW(
            std::ptr::null(),
            buffer.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            0,
            std::ptr::null(),
            std::ptr::null(),
            &startup,
            &mut process,
        ) != 0
        {
            CloseHandle(process.hProcess);
            CloseHandle(process.hThread);
        }
    }
}

/// Mugetsu: "Graphics" choices in the window's title-bar menu (right-click the title bar). A
/// different choice is saved and Mugetsu restarts with it. The title bar is drawn by Windows, so
/// this also works when the game itself shows nothing (e.g. a black screen on Vulkan); the other
/// settings are in the game's right-click menu (`gui::context_menu`).
mod graphics_menu {
    use crate::mugetsu::{self, Graphics};
    use std::sync::atomic::{AtomicIsize, Ordering};
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        AppendMenuW, CallWindowProcW, CheckMenuItem, GWLP_WNDPROC, GetSystemMenu, IDOK,
        MB_ICONQUESTION, MB_OKCANCEL, MF_BYCOMMAND, MF_CHECKED, MF_GRAYED, MF_SEPARATOR,
        MF_STRING, MF_UNCHECKED, MessageBoxW, SetWindowLongPtrW, WM_SYSCOMMAND, WNDPROC,
    };

    // System-menu command ids must keep their low 4 bits clear and stay below 0xF000.
    const AUTOMATIC: usize = 0x1000;
    const VULKAN: usize = 0x1010;
    const DX12: usize = 0x1020;
    const MORE: usize = 0x1040;

    static PREVIOUS_PROC: AtomicIsize = AtomicIsize::new(0);

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }

    pub fn install(hwnd: HWND) {
        if PREVIOUS_PROC.load(Ordering::Relaxed) != 0 {
            return;
        }
        let current = mugetsu::saved_graphics();
        let items = [
            (AUTOMATIC, "Graphics: Automatic (recommended)", current.is_none()),
            (VULKAN, "Graphics: Vulkan", current == Some(Graphics::Vulkan)),
            (DX12, "Graphics: DirectX 12", current == Some(Graphics::Dx12)),
        ];
        // SAFETY: `hwnd` is our live window, on its own thread; the strings outlive the calls.
        // The replaced window procedure is kept and called for everything else.
        unsafe {
            let menu = GetSystemMenu(hwnd, 0);
            if menu.is_null() {
                return;
            }
            AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
            for (id, text, checked) in items {
                let text = wide(text);
                let flags = MF_STRING | if checked { MF_CHECKED } else { 0 };
                AppendMenuW(menu, flags, id, text.as_ptr());
            }
            let text = wide("More settings: right-click the game");
            AppendMenuW(menu, MF_STRING | MF_GRAYED, MORE, text.as_ptr());
            let previous = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, window_proc as *const () as isize);
            PREVIOUS_PROC.store(previous, Ordering::Relaxed);
        }
    }

    unsafe extern "system" fn window_proc(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if message == WM_SYSCOMMAND {
            let choice = match wparam & 0xFFF0 {
                AUTOMATIC => Some(None),
                VULKAN => Some(Some(Graphics::Vulkan)),
                DX12 => Some(Some(Graphics::Dx12)),
                _ => None,
            };
            if let Some(choice) = choice {
                choose(hwnd, choice);
                return 0;
            }
        }
        // SAFETY: the previous procedure of this window, as `SetWindowLongPtrW` returned it.
        unsafe {
            let previous: WNDPROC = std::mem::transmute(PREVIOUS_PROC.load(Ordering::Relaxed));
            CallWindowProcW(previous, hwnd, message, wparam, lparam)
        }
    }

    /// The menu's check marks as the settings are now (also changed in the game's menu).
    pub(super) fn sync_checks(hwnd: HWND) {
        let current = mugetsu::saved_graphics();
        // SAFETY: `hwnd` is our live window; its system menu exists while it does.
        unsafe {
            let menu = GetSystemMenu(hwnd, 0);
            if !menu.is_null() {
                for (id, checked) in [
                    (AUTOMATIC, current.is_none()),
                    (VULKAN, current == Some(Graphics::Vulkan)),
                    (DX12, current == Some(Graphics::Dx12)),
                ] {
                    let state = if checked { MF_CHECKED } else { MF_UNCHECKED };
                    CheckMenuItem(menu, id as u32, MF_BYCOMMAND | state);
                }
            }
        }
    }

    pub(super) fn choose(hwnd: HWND, choice: Option<Graphics>) {
        if choice == mugetsu::saved_graphics() {
            return;
        }
        let name = choice.map_or("Automatic", Graphics::name);
        let text = wide(&format!(
            "Mugetsu will restart to use {name} graphics.\n\n\
             You can switch again any time: right-click the game or the title bar."
        ));
        let title = wide("Mugetsu");
        // SAFETY: both strings are NUL-terminated UTF-16 that outlive the call.
        let answer = unsafe {
            MessageBoxW(hwnd, text.as_ptr(), title.as_ptr(), MB_OKCANCEL | MB_ICONQUESTION)
        };
        if answer != IDOK {
            return;
        }
        mugetsu::save_graphics(choice);
        tracing::info!("Graphics changed to {name}; starting Mugetsu again");
        super::start_again();
        std::process::exit(0);
    }
}

fn hwnd_of(window: &winit::window::Window) -> Option<windows_sys::Win32::Foundation::HWND> {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    match window.window_handle().ok()?.as_raw() {
        RawWindowHandle::Win32(handle) => Some(handle.hwnd.get() as _),
        _ => None,
    }
}

pub fn add_graphics_menu(window: &winit::window::Window) {
    if let Some(hwnd) = hwnd_of(window) {
        graphics_menu::install(hwnd);
    }
}

/// Mugetsu: a graphics choice from the game's right-click menu (asks, then restarts).
pub fn choose_graphics(window: &winit::window::Window, choice: Option<crate::mugetsu::Graphics>) {
    if let Some(hwnd) = hwnd_of(window) {
        graphics_menu::choose(hwnd, choice);
    }
}

/// Mugetsu: the title-bar menu's check marks after a change in the game's right-click menu.
pub fn sync_menu_checks(window: &winit::window::Window) {
    if let Some(hwnd) = hwnd_of(window) {
        graphics_menu::sync_checks(hwnd);
    }
}

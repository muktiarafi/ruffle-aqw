use super::dialogs::export_bundle_dialog::ExportBundleDialogConfiguration;
use super::{DialogDescriptor, FilePicker};
use crate::backends::DesktopUiBackend;
use crate::custom_event::RuffleEvent;
use crate::gui::movie::{MovieView, MovieViewRenderer};
use crate::gui::theme::ThemeController;
use crate::gui::{MENU_HEIGHT, RuffleGui};
use crate::player::{LaunchOptions, PlayerController};
use crate::preferences::GlobalPreferences;
use anyhow::anyhow;
use egui::{Context, FontData, FontDefinitions, ViewportId};
use fontdb::{Database, Family, Query, Source};
use ruffle_core::events::{ImeCursorArea, ImePurpose};
use ruffle_core::{Player, PlayerEvent};
use ruffle_frontend_utils::content::ContentDescriptor;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::present::PresentThread;
use ruffle_render_wgpu::utils::{format_list, get_backend_names};
use std::any::Any;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use url::Url;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::{Key, NamedKey};
use winit::window::{ImePurpose as WinitImePurpose, Theme, Window};

/// AQW build: the window's surface and how it's configured, shared with the present thread
/// (see `present_frame`).
struct SurfaceSlot {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

/// AQW build: a window frame to draw and present (see `present_frame`).
struct GuiFrame {
    primitives: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    screen_descriptor: egui_wgpu::ScreenDescriptor,
    /// The movie's picture (its view's texture), if a movie is open.
    movie: Option<wgpu::BindGroup>,
}

/// Integration layer connecting wgpu+winit to egui.
pub struct GuiController {
    descriptors: Arc<Descriptors>,
    egui_winit: egui_winit::State,
    /// AQW build: used by the present thread (see `present_frame`).
    egui_renderer: Arc<Mutex<egui_wgpu::Renderer>>,
    gui: RuffleGui,
    window: Arc<Window>,
    last_update: Instant,
    repaint_after: Duration,
    surface: Arc<Mutex<SurfaceSlot>>,
    movie_view_renderer: Arc<MovieViewRenderer>,
    /// AQW build: frames are submitted and shown from here (see
    /// `ruffle_render_wgpu::present`); none when that's off (`PresentThread::enabled`).
    present: Option<Arc<PresentThread>>,
    /// AQW build: with the render thread, what draws the movie's texture, and the state of
    /// the movie's target it was read at (see `render`).
    movie_binds: Option<(u64, wgpu::BindGroup)>,
    // Note that `window.get_inner_size` can change at any point on x11, even between two lines of code.
    // Use this instead.
    size: PhysicalSize<u32>,
    /// If this is set, we should not render the main menu.
    no_gui: bool,
    /// AQW build: whether egui drew anything (menu bar, dialog, context menu, debug UI) in the
    /// last frame. Without any of it, a GUI-only repaint would present the same picture again.
    ui_visible: bool,
    theme_controller: ThemeController,
    #[cfg(feature = "tracy_images")]
    tracy_frame_captures: crate::tracy::FrameCapturesHolder,
}

impl GuiController {
    pub fn new(
        window: Arc<Window>,
        event_loop: EventLoopProxy<RuffleEvent>,
        preferences: GlobalPreferences,
        font_database: &Database,
        initial_movie_url: Option<Url>,
        no_gui: bool,
    ) -> anyhow::Result<Self> {
        let wanted: wgpu::Backends = preferences.graphics_backends().into();
        // AQW build: a driver crash from here until shortly after the first frame counts as
        // the Vulkan driver crashing at startup (see `mugetsu`).
        crate::mugetsu::graphics_startup_begin(
            wanted.contains(wgpu::Backends::VULKAN),
            wanted.contains(wgpu::Backends::DX12),
        );
        let open = |wanted: wgpu::Backends| -> anyhow::Result<_> {
            let (instance, backend) = select_wgpu_backend(wanted, &window)?;
            // (The one that is really being started: another one is used where the wanted
            // one is not there at all.)
            crate::mugetsu::graphics_startup_begin(
                backend.contains(wgpu::Backends::VULKAN),
                backend.contains(wgpu::Backends::DX12),
            );
            #[cfg(windows)]
            crate::windows::test_native_crash();
            let surface = instance.create_surface(window.clone())?;
            let (adapter, device, queue) =
                futures::executor::block_on(request_adapter_and_device(
                    backend,
                    &instance,
                    Some(&surface),
                    preferences.graphics_power_preference().into(),
                ))
                .map_err(|e| anyhow!(e.to_string()))?;
            Ok((instance, surface, adapter, device, queue))
        };
        // AQW build: DirectX 12 is the default on Windows. Where it is there but cannot be
        // started (an old graphics chip or driver), Vulkan is tried before giving up.
        // (`RUFFLE_TEST_DX12_FAILS`, testing: DirectX 12 is taken as not startable.)
        let first = if wanted == wgpu::Backends::DX12
            && std::env::var_os("RUFFLE_TEST_DX12_FAILS").is_some()
        {
            Err(anyhow!("RUFFLE_TEST_DX12_FAILS is set"))
        } else {
            open(wanted)
        };
        let (instance, surface, adapter, device, queue) = match first {
            Err(error) if wanted == wgpu::Backends::DX12 => {
                tracing::warn!("DirectX 12 could not be started ({error}); trying Vulkan");
                crate::mugetsu::graphics_startup_begin(true, false);
                open(wgpu::Backends::VULKAN)?
            }
            opened => opened?,
        };
        let adapter_info = adapter.get_info();
        crate::mugetsu::set_adapter_ids(adapter_info.vendor, adapter_info.device);
        #[cfg(not(windows))]
        crate::mugetsu::set_video_memory_adapter(adapter.clone());
        // Which drawing speed-ups are switched off (for test logs).
        let off = crate::mugetsu::drawing_switches_off();
        if !off.is_empty() {
            tracing::info!("Drawing speed-ups switched off: {off:?}");
        }
        ruffle_render_wgpu::set_transient_textures_enabled(
            !crate::mugetsu::exe_flag("no-transient-memory"),
        );
        tracing::info!(
            "Using graphics API {} on {} (type: {:?})",
            adapter_info.backend.to_str(),
            adapter_info.name,
            adapter_info.device_type
        );
        if wanted.contains(wgpu::Backends::VULKAN) && adapter_info.backend != wgpu::Backend::Vulkan
        {
            aqw_warn_vulkan_fallback(&preferences, adapter_info.backend, &window, &event_loop);
        } else if wanted == wgpu::Backends::DX12 && adapter_info.backend != wgpu::Backend::Dx12 {
            // AQW build: Vulkan runs as well, so the title says it and nothing is asked.
            let name = match adapter_info.backend {
                wgpu::Backend::Vulkan => "Vulkan",
                wgpu::Backend::Gl => "OpenGL, slower",
                other => other.to_str(),
            };
            tracing::warn!("DirectX 12 is not available; using {name}");
            crate::mugetsu::add_title_note(format!("{name}: DirectX 12 not available"));
            window.set_title(&crate::window_title());
        }
        crate::mugetsu::check_gpu_choice(
            window.clone(),
            event_loop.clone(),
            &adapter_info,
            preferences.graphics_power_preference().into(),
        );
        // AQW build: detailed shapes drawn tiny are tessellated for at least 32 px, as those
        // tessellations stalled a laptop's CPU; with a dedicated card they keep the old 8 px
        // (see core's `tessellation_cache::set_min_size`).
        let shape_min_px = ruffle_core::set_shape_min_px(
            if adapter_info.device_type == wgpu::DeviceType::DiscreteGpu {
                8.0
            } else {
                32.0
            },
        );
        tracing::info!("Detailed shapes are tessellated for at least {shape_min_px} px");
        crate::mugetsu::spawn_update_check(window.clone(), event_loop.clone());
        #[cfg(windows)]
        if crate::mugetsu::plays_aqw() {
            crate::windows::add_graphics_menu(&window, event_loop.clone());
        }
        let preferred_formats = [
            // by egui
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ];
        let supported_formats = surface.get_capabilities(&adapter).formats;
        let surface_format = preferred_formats
            .iter()
            .find(|format| supported_formats.contains(format))
            .copied()
            .unwrap_or_else(|| {
                supported_formats
                    .first()
                    .copied()
                    .expect("At least one format should be supported")
            });
        tracing::info!("Using surface format {:?}", surface_format);
        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width,
            height: size.height,
            present_mode: Default::default(),
            desired_maximum_frame_latency: 2,
            alpha_mode: Default::default(),
            view_formats: Default::default(),
        };
        surface.configure(&device, &surface_config);
        let descriptors = Descriptors::new(instance, adapter, device, queue);
        let egui_ctx = Context::default();
        // The rest of the desktop UI (menu bar offset, movie viewport, egui-wgpu
        // screen descriptor) assumes egui's pixels_per_point equals the window
        // scale factor, so egui's own UI zoom would break the layout.
        egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);

        let theme_controller = futures::executor::block_on(ThemeController::new(
            window.clone(),
            preferences.clone(),
            egui_ctx.clone(),
        ));
        let mut egui_winit = egui_winit::State::new(
            egui_ctx,
            ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            None,
        );
        egui_winit.set_max_texture_side(descriptors.limits.max_texture_dimension_2d as usize);

        let movie_view_renderer = Arc::new(MovieViewRenderer::new(
            &descriptors.device,
            surface_format,
            window.fullscreen().is_none() && !no_gui,
            size.height,
            window.scale_factor(),
        ));
        let egui_renderer = egui_wgpu::Renderer::new(
            &descriptors.device,
            surface_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );
        let descriptors = Arc::new(descriptors);
        let gui = RuffleGui::new(
            Arc::downgrade(&window),
            event_loop,
            initial_movie_url.map(|url| ContentDescriptor {
                url,
                root_content_path: None,
            }),
            LaunchOptions::from(&preferences),
            preferences.clone(),
        );
        let system_fonts = load_system_fonts(font_database, preferences.language());
        egui_winit.egui_ctx().set_fonts(system_fonts);

        egui_extras::install_image_loaders(egui_winit.egui_ctx());

        #[cfg(feature = "tracy_images")]
        let tracy_frame_captures = crate::tracy::FrameCapturesHolder::new(&descriptors.device);

        Ok(Self {
            descriptors,
            egui_winit,
            egui_renderer: Arc::new(Mutex::new(egui_renderer)),
            gui,
            window,
            last_update: Instant::now(),
            repaint_after: Duration::ZERO,
            surface: Arc::new(Mutex::new(SurfaceSlot {
                surface,
                config: surface_config,
            })),
            movie_view_renderer,
            present: PresentThread::enabled().then(|| PresentThread::new("present")),
            movie_binds: None,
            size,
            no_gui,
            ui_visible: true,
            theme_controller,
            #[cfg(feature = "tracy_images")]
            tracy_frame_captures,
        })
    }

    pub fn set_theme(&self, theme: Theme) {
        self.theme_controller.set_theme(theme);
    }

    pub fn descriptors(&self) -> &Arc<Descriptors> {
        &self.descriptors
    }

    pub fn file_picker(&self) -> FilePicker {
        self.gui.dialogs.file_picker()
    }

    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.size = size;
            self.reconfigure_surface();
        }
    }

    /// AQW build: waits until the present thread has done everything sent to it so far (see
    /// `settle_before_resize` in the app).
    pub fn settle_present(&self) {
        if let Some(present) = &self.present {
            present.wait_for(present.run(Box::new(|| {})));
        }
    }

    /// AQW build: whether frames sent to the window wait for the monitor's refresh (v-sync,
    /// as always before) or go out at once ("uncapped" frame pacing, see
    /// `mugetsu::uncapped`): then the first of immediate (tearing allowed, what a
    /// variable-refresh monitor follows), mailbox (the newest frame at each refresh) and
    /// v-sync that the system has. The caller lets everything on its way through first, as
    /// before a resize. Says what it got.
    pub fn set_vsync(&self, vsync: bool) -> &'static str {
        let (mode, got) = if vsync {
            (wgpu::PresentMode::Fifo, "in step with the monitor (v-sync)")
        } else {
            let has = lock(&self.surface)
                .surface
                .get_capabilities(&self.descriptors.adapter)
                .present_modes;
            let got = if has.contains(&wgpu::PresentMode::Immediate) {
                "at once (immediate)"
            } else if has.contains(&wgpu::PresentMode::Mailbox) {
                "at once, the newest at each refresh (mailbox)"
            } else {
                "in step with the monitor: this system has no other way"
            };
            (wgpu::PresentMode::AutoNoVsync, got)
        };
        let changed = {
            let mut slot = lock(&self.surface);
            let changed = slot.config.present_mode != mode;
            slot.config.present_mode = mode;
            changed
        };
        if changed {
            self.reconfigure_surface();
        }
        got
    }

    pub fn reconfigure_surface(&self) {
        {
            let mut slot = lock(&self.surface);
            let SurfaceSlot { surface, config } = &mut *slot;
            config.width = self.size.width;
            config.height = self.size.height;
            // AQW build: (with the render thread: nothing submits meanwhile.)
            let _gate = ruffle_render_wgpu::render_thread::reconfiguring();
            surface.configure(&self.descriptors.device, config);
        }
        self.movie_view_renderer.update_resolution(
            &self.descriptors,
            self.window.fullscreen().is_none() && !self.no_gui,
            self.size.height,
            self.window.scale_factor(),
        );
    }

    #[must_use]
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::Resized(size) = &event {
            self.resize(*size);
        }

        if let WindowEvent::ThemeChanged(theme) = &event {
            self.set_theme(*theme);
        }

        if matches!(
            &event,
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    logical_key: Key::Named(NamedKey::Tab),
                    ..
                },
                ..
            }
        ) {
            // Prevent egui from consuming the Tab key.
            return false;
        }

        let response = self.egui_winit.on_window_event(&self.window, event);
        // AQW build: egui asks for a repaint on every mouse move. When it shows nothing, the
        // repaint would only present the movie's last frame again (extra presents that frame
        // counters and frame generation tools such as Lossless Scaling count as frames); the
        // input still reaches egui with the next movie frame.
        if response.repaint && self.wants_redraws() {
            self.window.request_redraw();
        }
        response.consumed
    }

    pub fn close_movie(&mut self, player: &mut PlayerController) {
        player.destroy();
        self.gui.on_player_destroyed();
    }

    pub fn create_movie(
        &mut self,
        player: &mut PlayerController,
        opt: LaunchOptions,
        content_descriptor: ContentDescriptor,
    ) {
        tracing::info!("Opening {}", content_descriptor.describe());

        self.close_movie(player);
        let movie_view = MovieView::new(
            self.movie_view_renderer.clone(),
            &self.descriptors.device,
            crate::mugetsu::drawn_len(self.size.width),
            crate::mugetsu::drawn_len(self.size.height),
            #[cfg(feature = "tracy_images")]
            self.tracy_frame_captures.clone(),
        );
        player.create(&opt, &content_descriptor, movie_view);
        // AQW build: the movie's frames are submitted and shown from the present thread (see
        // `ruffle_render_wgpu::present`).
        if let Some(present) = &self.present
            && let Some(mut player) = player.get()
        {
            ruffle_render_wgpu::render_thread::with_wgpu_backend(
                player.renderer_mut(),
                |renderer: &mut WgpuRenderBackend<MovieView>| {
                    renderer.set_present_thread(present.clone())
                },
            );
        }
        self.movie_binds = None;
        self.gui.on_player_created(
            opt,
            content_descriptor,
            player
                .get()
                .expect("Player must exist after being created."),
        );
    }

    pub fn height_offset(&self) -> f64 {
        if self.window.fullscreen().is_some() || self.no_gui {
            0.0
        } else {
            MENU_HEIGHT as f64 * self.window.scale_factor()
        }
    }

    pub fn window_to_movie_position(&self, position: PhysicalPosition<f64>) -> (f64, f64) {
        // AQW build: (the movie's own pixels are fewer than the window's in Performance
        // mode, see `mugetsu::Performance`.)
        let scale = crate::mugetsu::render_scale();
        let x = position.x * scale;
        let y = (position.y - self.height_offset()) * scale;
        (x, y)
    }

    pub fn movie_to_window_position(&self, x: f64, y: f64) -> PhysicalPosition<f64> {
        let scale = crate::mugetsu::render_scale();
        let y = y / scale + self.height_offset();
        PhysicalPosition::new(x / scale, y)
    }

    /// AQW build: runs the UI on this thread, then draws and presents the window frame: on the
    /// present thread, after the movie's own submit (see `ruffle_render_wgpu::present`), or
    /// here when that's switched off (`present_frame` either way).
    pub fn render(&mut self, mut player: Option<MutexGuard<Player>>) {
        #[cfg(feature = "tracy_images")]
        if player.is_none() {
            self.tracy_frame_captures
                .set_target(&self.descriptors.device, None);
        }

        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let show_menu = self.window.fullscreen().is_none() && !self.no_gui;
        let mut counter_shapes = 0;
        let counter_top = self.height_offset();
        let mut full_output = self.egui_winit.egui_ctx().run_ui(raw_input, |ui| {
            self.gui.update(
                self.egui_winit.egui_ctx(),
                ui,
                show_menu,
                player.as_deref_mut(),
                if show_menu {
                    MENU_HEIGHT as f64 * self.window.scale_factor()
                } else {
                    0.0
                },
            );
            if crate::mugetsu::show_fps() {
                counter_shapes = fps_counter(ui.ctx(), counter_top);
            }
        });
        // AQW build: the FPS counter (Mugetsu's settings) is redrawn with the movie's frames
        // only, so on its own it isn't UI that wants repaints (see `wants_redraws`).
        self.ui_visible = full_output.shapes.len() > counter_shapes;
        self.repaint_after = full_output
            .viewport_output
            .get(&ViewportId::ROOT)
            .expect("Root viewport must exist")
            .repaint_delay;

        // If we're not in a UI, tell egui which cursor we prefer to use instead
        if !self.egui_winit.egui_ctx().egui_wants_pointer_input()
            && let Some(player) = player.as_deref()
        {
            full_output.platform_output.cursor_icon =
                <dyn Any>::downcast_ref::<DesktopUiBackend>(player.ui())
                    .unwrap_or_else(|| panic!("UI Backend should be DesktopUiBackend"))
                    .cursor();
        }
        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);

        let clipped_primitives = self
            .egui_winit
            .egui_ctx()
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let scale_factor = self.window.scale_factor() as f32;
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.size.width, self.size.height],
            pixels_per_point: scale_factor,
        };

        // (From upstream.) Debug tools draw over the movie and can modify it. While any of
        // them is open, show that right away, instead of waiting for the movie to render on
        // its own. This happens after updating the GUI, so that the changes made in this pass
        // are visible.
        if let Some(player) = player.as_deref_mut()
            && player.debug_ui().has_open_windows()
        {
            player.render();
        }

        // AQW build: with the render thread, what draws the movie's texture is kept here
        // for as long as the movie's target can't have changed (a resize, a new movie), so
        // that asking for it doesn't wait for the picture being recorded (see
        // `ruffle_render_wgpu::render_thread`).
        let movie = player.as_deref_mut().and_then(|player| {
            use ruffle_render_wgpu::render_thread;
            let fresh = |player: &mut Player| {
                render_thread::with_wgpu_backend(
                    player.renderer_mut(),
                    |renderer: &mut WgpuRenderBackend<MovieView>| renderer.target().bind_group(),
                )
            };
            let Some(state) = render_thread::target_state::<MovieView>(player.renderer()) else {
                return fresh(player);
            };
            if self.movie_binds.as_ref().is_none_or(|(known, _)| *known != state) {
                let binds = fresh(player)?;
                // (Reading it counted as a change of state itself.)
                let state = render_thread::target_state::<MovieView>(player.renderer())?;
                self.movie_binds = Some((state, binds));
            }
            self.movie_binds.as_ref().map(|(_, binds)| binds.clone())
        });
        let frame = GuiFrame {
            primitives: clipped_primitives,
            textures_delta: full_output.textures_delta,
            screen_descriptor,
            movie,
        };
        #[cfg(feature = "tracy_images")]
        let tracy_frame_captures = self.tracy_frame_captures.clone();
        if let Some(present) = &self.present {
            let descriptors = self.descriptors.clone();
            let surface = self.surface.clone();
            let egui_renderer = self.egui_renderer.clone();
            let movie_view_renderer = self.movie_view_renderer.clone();
            let window = self.window.clone();
            let job: ruffle_render_wgpu::present::Job = Box::new(move || {
                present_frame(
                    &descriptors,
                    &surface,
                    &egui_renderer,
                    &movie_view_renderer,
                    &window,
                    frame,
                    #[cfg(feature = "tracy_images")]
                    tracy_frame_captures,
                )
            });
            // AQW build: with the render thread, the movie's own submit reaches the present
            // thread from there: the window's frame is sent after it, through the same
            // thread, or it would show the picture before (see
            // `ruffle_render_wgpu::render_thread`).
            let present = present.clone();
            let send: ruffle_render_wgpu::present::Job = Box::new(move || {
                present.run(job);
            });
            match player.as_deref_mut() {
                Some(player) => ruffle_render_wgpu::render_thread::run_after_picture::<MovieView>(
                    player.renderer_mut(),
                    send,
                ),
                None => send(),
            }
        } else {
            present_frame(
                &self.descriptors,
                &self.surface,
                &self.egui_renderer,
                &self.movie_view_renderer,
                &self.window,
                frame,
                #[cfg(feature = "tracy_images")]
                tracy_frame_captures,
            );
        }
    }

    pub fn show_context_menu(
        &mut self,
        menu: Vec<ruffle_core::ContextMenuItem>,
        settings: Option<crate::mugetsu::MenuSettings>,
        close_event: PlayerEvent,
    ) {
        self.gui.show_context_menu(menu, settings, close_event);
        self.window.request_redraw();
    }

    pub fn is_context_menu_visible(&self) -> bool {
        self.gui.is_context_menu_visible()
    }

    /// AQW build: see `RuffleGui::ask_graphics_restart`.
    pub fn ask_graphics_restart(&mut self, choice: Option<crate::mugetsu::Graphics>) {
        self.gui.ask_graphics_restart(choice);
        self.window.request_redraw();
    }

    /// AQW build: see `RuffleGui::offer_interpolation_restart`.
    pub fn offer_interpolation_restart(&mut self, wanted: bool) {
        self.gui.offer_interpolation_restart(wanted);
        self.window.request_redraw();
    }

    /// AQW build: see `RuffleGui::show_notice`.
    pub fn show_notice(&mut self, notice: crate::mugetsu::Notice) {
        self.gui.show_notice(notice);
        self.window.request_redraw();
    }

    /// AQW build: whether GUI-only repaints are useful: egui shows something, or is about to.
    pub fn wants_redraws(&self) -> bool {
        self.ui_visible || self.gui.is_context_menu_visible()
    }

    pub fn needs_render(&self) -> bool {
        Instant::now().duration_since(self.last_update) >= self.repaint_after
    }

    pub fn show_open_dialog(&mut self) {
        self.gui.dialogs.open_file_advanced()
    }

    pub fn open_dialog(&mut self, dialog_event: DialogDescriptor) {
        self.gui.dialogs.open_dialog(dialog_event);
        self.window.request_redraw();
    }

    pub fn set_ime_allowed(&self, allowed: bool) {
        self.window.set_ime_allowed(allowed);
    }

    pub fn set_ime_purpose(&self, purpose: ImePurpose) {
        self.window.set_ime_purpose(match purpose {
            ImePurpose::Standard => WinitImePurpose::Normal,
            ImePurpose::Password => WinitImePurpose::Password,
        });
    }

    pub fn set_ime_cursor_area(&self, cursor_area: ImeCursorArea) {
        self.window.set_ime_cursor_area(
            self.movie_to_window_position(cursor_area.x, cursor_area.y),
            PhysicalSize::new(cursor_area.width, cursor_area.height),
        );
    }

    pub fn export_bundle(&mut self) {
        let Some(content_descriptor) = self.gui.dialogs.saved_content_descriptor() else {
            return;
        };

        let launch_options = self.gui.dialogs.saved_launch_options();
        let player_options = launch_options.player.clone();
        self.gui
            .dialogs
            .open_dialog(DialogDescriptor::ExportBundle(Box::new(
                ExportBundleDialogConfiguration::new(content_descriptor, player_options),
            )));
        self.gui.on_player_destroyed();
    }
}

fn select_wgpu_backend(
    preferred_backends: wgpu::Backends,
    window: &Arc<Window>,
) -> anyhow::Result<(wgpu::Instance, wgpu::Backends)> {
    for backend in preferred_backends.iter() {
        if let Some(instance) = try_wgpu_backend(backend, window.clone()) {
            tracing::info!(
                "Using preferred backend {}",
                format_list(&get_backend_names(backend), "and")
            );
            return Ok((instance, backend));
        }
    }

    tracing::warn!(
        "Preferred backend(s) of {} not available; falling back to any",
        format_list(&get_backend_names(preferred_backends), "or")
    );

    // AQW build: try the native APIs before OpenGL (plain bit order tried GL before DX12,
    // and wgpu's GL backend is much slower on Windows). Vulkan too: it is what DirectX 12,
    // the default on Windows, falls back to.
    let remaining = wgpu::Backends::all() - preferred_backends;
    let native = [
        wgpu::Backends::DX12,
        wgpu::Backends::VULKAN,
        wgpu::Backends::METAL,
        wgpu::Backends::GL,
    ];
    let rest = native
        .iter()
        .fold(remaining, |rest, backend| rest - *backend);
    let ordered = native
        .into_iter()
        .filter(|b| remaining.contains(*b))
        .chain(rest.iter());
    for backend in ordered {
        if let Some(instance) = try_wgpu_backend(backend, window.clone()) {
            tracing::info!(
                "Using fallback backend {}",
                format_list(&get_backend_names(backend), "and")
            );
            return Ok((instance, backend));
        }
    }

    Err(anyhow!(
        "No compatible graphics backends of any kind were available"
    ))
}

fn try_wgpu_backend(backend: wgpu::Backends, window: Arc<Window>) -> Option<wgpu::Instance> {
    let instance = create_wgpu_instance(
        backend,
        wgpu::BackendOptions::default(),
        Some(Box::new(window)),
    );
    if futures::executor::block_on(instance.enumerate_adapters(backend)).is_empty() {
        None
    } else {
        Some(instance)
    }
}

// Load fallback fonts
/// AQW build: draws the window frame (the movie's picture, then the UI) into the surface and
/// presents it. From the present thread (after the movie's submit, which it ordered before
/// this) or the main thread (see `GuiController::render`).
fn present_frame(
    descriptors: &Descriptors,
    surface: &Mutex<SurfaceSlot>,
    egui_renderer: &Mutex<egui_wgpu::Renderer>,
    movie_view_renderer: &MovieViewRenderer,
    window: &Window,
    mut frame: GuiFrame,
    #[cfg(feature = "tracy_images")] tracy_frame_captures: crate::tracy::FrameCapturesHolder,
) {
    // The UI's texture changes first (a font atlas, say): a frame that can't be drawn must not
    // lose them, or the menus would show missing or garbled text until egui remade them.
    let mut egui_renderer = lock(egui_renderer);
    for (id, image_deltas) in frame.textures_delta.set.drain() {
        for image_delta in &image_deltas {
            egui_renderer.update_texture(
                &descriptors.device,
                &descriptors.queue,
                id,
                image_delta,
            );
        }
    }
    let mut slot = lock(surface);
    // Laid out for another size: the window was resized after this frame was queued (the
    // resize reconfigured the surface under this lock). Skip it; the next frame is laid out
    // for the new size. Drawn, its menu would fall outside the surface (a wgpu validation
    // error: "Scissor Rect ... is not contained in the render target", fatal).
    if frame.screen_descriptor.size_in_pixels != [slot.config.width, slot.config.height] {
        free_textures(&mut egui_renderer, &mut frame);
        return;
    }
    let surface_texture = match slot.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
        wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
            // The acquired texture is still usable - no reason to waste
            // this frame. The surface should ideally be reconfigured
            // before the next acquire; usually a resize event arrives and
            // does that, but not always (e.g. on a display transform
            // change) - we accept staying suboptimal in that case for now.
            surface_texture
        }
        state @ (wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated) => {
            // Reconfigure the surface if lost or outdated.
            // Some sources suggest ignoring `Outdated` and waiting for the next frame,
            // but I suspect this advice is related explicitly to resizing,
            // because the future resize event will reconfigure the surface.
            // However, resizing is not the only possible reason for the surface
            // to become outdated (resolution / refresh rate change, some internal
            // platform-specific reasons, wgpu bugs?).
            // Testing on Vulkan shows that reconfiguring the surface works in that case.
            tracing::warn!("Surface became unavailable: {:?}, reconfiguring", state);
            let SurfaceSlot { surface, config } = &mut *slot;
            // AQW build: (with the render thread: nothing submits meanwhile.)
            let _gate = ruffle_render_wgpu::render_thread::reconfiguring();
            surface.configure(&descriptors.device, config);
            free_textures(&mut egui_renderer, &mut frame);
            return;
        }
        wgpu::CurrentSurfaceTexture::Timeout => {
            // Acquiring the frame took too long (a full second, as of wgpu 29).
            // The best thing we can do is skip a frame and wait.
            tracing::warn!("Surface acquisition timed out, skipping a frame");
            free_textures(&mut egui_renderer, &mut frame);
            return;
        }
        wgpu::CurrentSurfaceTexture::Occluded => {
            // The window is occluded (e.g. minimized or fully covered by
            // another window), so there is no point in drawing anything.
            // We normally don't even get here, as `WindowEvent::Occluded`
            // makes us skip rendering entirely - but the two can race, and
            // not every platform reports occlusion through winit.
            // Unlike `Timeout`, this returns immediately, so it would repeat
            // at the movie's frame rate: keep it quiet, it's not an error.
            tracing::trace!("Surface is occluded, skipping a frame");
            free_textures(&mut egui_renderer, &mut frame);
            return;
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            // Generic error, not much we can do.
            panic!("wgpu: Acquiring a texture failed with a validation error");
        }
    };

    let mut encoder = descriptors
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui encoder"),
        });

    let mut command_buffers = egui_renderer.update_buffers(
        &descriptors.device,
        &descriptors.queue,
        &mut encoder,
        &frame.primitives,
        &frame.screen_descriptor,
    );

    {
        let surface_view = surface_texture.texture.create_view(&Default::default());

        let mut render_pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                label: Some("egui_render"),
                ..Default::default()
            })
            .forget_lifetime();

        if let Some(movie) = &frame.movie {
            crate::gui::movie::blit(movie_view_renderer, movie, &mut render_pass);
        }

        egui_renderer.render(&mut render_pass, &frame.primitives, &frame.screen_descriptor);

        #[cfg(feature = "tracy_images")]
        {
            drop(render_pass);
            tracy_frame_captures.capture_frame(&descriptors.device, &mut encoder);
        }
    }

    command_buffers.push(encoder.finish());
    descriptors.queue.submit(command_buffers);

    // Free textures only after submitting the command buffer that may still
    // reference them.  Destroying a texture before its usage is submitted
    // causes a wgpu validation panic.
    free_textures(&mut egui_renderer, &mut frame);
    drop(egui_renderer);

    window.pre_present_notify();
    descriptors.queue.present(surface_texture);
    crate::mugetsu::frame_presented();
    #[cfg(feature = "tracy")]
    tracing_tracy::client::frame_mark();
    #[cfg(feature = "tracy_images")]
    tracy_frame_captures.finish_frame();
}

/// Frees the UI textures `frame` no longer uses.
fn free_textures(egui_renderer: &mut egui_wgpu::Renderer, frame: &mut GuiFrame) {
    for id in frame.textures_delta.free.drain() {
        egui_renderer.free_texture(&id);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn load_system_fonts(
    font_database: &Database,
    locale: unic_langid::LanguageIdentifier,
) -> egui::FontDefinitions {
    let mut fd: FontDefinitions = egui::FontDefinitions::default();

    let lang = locale.language.as_str();
    let is_ja = lang == "ja";
    let is_ko = lang == "ko";
    let is_zh = lang == "zh";
    let is_sc = is_zh && locale.to_string().as_str() == "zh-CN";
    let is_tc = is_zh && !is_sc;

    let mut queries: PrioritizedQueries = Vec::new();

    // The main font
    queries.push((1, vec![Family::SansSerif]));

    // Pan-CJK fonts
    queries.push((
        2,
        vec![
            Family::Name("Noto Sans CJK"),     // Open font
            Family::Name("Source Han Sans"),   // Open font, same as Noto Sans CJK
            Family::Name("WenQuanYi Zen Hei"), // Open font
            Family::Name("Arial Unicode MS"),  // MacOS
        ],
    ));

    // Korean
    queries.push((
        3 + if is_ko { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK KR"), // Open font
            Family::Name("Malgun Gothic"),    // Windows
        ],
    ));

    // Japanese
    queries.push((
        3 + if is_ja { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK JP"), // Open font
            Family::Name("MS UI Gothic"),     // Windows
        ],
    ));

    // Chinese Simplified
    queries.push((
        3 + if is_sc { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK SC"), // Open font
            Family::Name("Microsoft YaHei"),  // Windows
        ],
    ));

    // Chinese Traditional
    queries.push((
        3 + if is_tc { 0 } else { 1 },
        vec![
            Family::Name("Noto Sans CJK TC"),   // Open font
            Family::Name("Microsoft JhengHei"), // Windows
        ],
    ));

    // Hebrew
    queries.push((
        4,
        vec![
            Family::Name("Noto Sans Hebrew"), // Open font
            Family::Name("Tahoma"),           // Windows
        ],
    ));

    // Arabic
    queries.push((
        5,
        vec![
            Family::Name("Noto Sans Arabic"), // Open font
            Family::Name("Tahoma"),           // Windows
        ],
    ));

    // Thai
    queries.push((
        6,
        vec![
            Family::Name("Noto Sans Thai"), // Open font
            Family::Name("Tahoma"),         // Windows
        ],
    ));

    register_family(
        font_database,
        &mut fd,
        egui::FontFamily::Proportional,
        queries,
    );

    fd
}

type FamilyQuery<'a> = Vec<Family<'a>>;
type PrioritizedQueries<'a> = Vec<(usize, FamilyQuery<'a>)>;

fn register_family(
    font_database: &Database,
    fd: &mut FontDefinitions,
    family: egui::FontFamily,
    mut queries: PrioritizedQueries<'_>,
) {
    queries.sort_by_key(|(priority, _)| *priority);
    for (_, query) in queries {
        register_family_font(font_database, fd, family.clone(), &query);
    }
}

fn register_family_font(
    font_database: &Database,
    fd: &mut FontDefinitions,
    family: egui::FontFamily,
    query: &FamilyQuery<'_>,
) {
    let (name, fontdata) = match load_system_font(font_database, query) {
        Ok((name, fontdata)) => (name, fontdata),
        Err(e) => {
            tracing::warn!("Failed to register {query:?} as {family}: {e}");
            return;
        }
    };

    tracing::debug!("Registering font {name} as {family}");

    fd.font_data.insert(name.clone(), fontdata.into());
    fd.families.entry(family).or_default().push(name);
}

fn load_system_font(
    font_database: &Database,
    families: &Vec<Family<'_>>,
) -> anyhow::Result<(String, FontData)> {
    let system_unicode_fonts = Query {
        families,
        ..Query::default()
    };

    let id = font_database
        .query(&system_unicode_fonts)
        .ok_or(anyhow!("no unicode fonts found!"))?;
    let (name, src, index) = font_database
        .face(id)
        .map(|f| (f.post_script_name.clone(), f.source.clone(), f.index))
        .expect("id not found in font database");

    let mut fontdata = match src {
        Source::File(path) | Source::SharedFile(path, _) => {
            let data = mmap_system_font(&path)?;

            // egui accepts only static data, so we have to leak mmapped fonts.
            // This is acceptable, as we're doing it only once.
            let data = Box::leak(Box::new(data));

            egui::FontData::from_static(data)
        }
        Source::Binary(bin) => {
            let data = bin.as_ref().as_ref().to_vec();
            egui::FontData::from_owned(data)
        }
    };
    fontdata.index = index;

    Ok((name, fontdata))
}

fn mmap_system_font(path: &Path) -> anyhow::Result<memmap2::Mmap> {
    let file = File::open(path).map_err(|e| anyhow!("Couldn't open font file at {path:?}: {e}"))?;

    // SAFETY: We have to assume that the font file won't change.
    // This assumption is realistic, as we're using system fonts only.
    let mmap = unsafe { memmap2::Mmap::map(&file) };

    let mmap = mmap.map_err(|e| anyhow!("Failed to mmap font file at {path:?}: {e}"))?;
    Ok(mmap)
}

/// AQW build: Vulkan was asked for (Mugetsu always does) but this PC doesn't have it, so Ruffle
/// fell back to another API. Says so in the window title, and once in the game window (a
/// marker file in the config directory remembers it was shown). DirectX 12 runs about as well
/// as Vulkan since Mugetsu 1.1; OpenGL is much slower.
fn aqw_warn_vulkan_fallback(
    preferences: &GlobalPreferences,
    backend: wgpu::Backend,
    window: &Window,
    events: &EventLoopProxy<RuffleEvent>,
) {
    let (name, slower) = match backend {
        wgpu::Backend::Dx12 => ("DirectX 12", false),
        wgpu::Backend::Gl => ("OpenGL", true),
        other => (other.to_str(), true),
    };
    tracing::warn!("Vulkan is not available; using {name}");
    crate::mugetsu::add_title_note(if slower {
        format!("{name}, slower: Vulkan not available")
    } else {
        format!("{name}: Vulkan not available")
    });
    window.set_title(&crate::window_title());

    let marker = preferences.cli.config.join("mugetsu-vulkan-warning-shown");
    if marker.exists() {
        return;
    }
    let _ = std::fs::create_dir_all(&preferences.cli.config);
    let _ = std::fs::write(&marker, name);
    // Shown in the game window once it's up (see `gui::prompt`); the game keeps loading.
    let _ = events.send_event(RuffleEvent::MugetsuNotice(
        crate::mugetsu::Notice::VulkanUnavailable { name, slower },
    ));
}

/// AQW build: draws the FPS counter (a small badge in the top-left corner of the game); returns
/// how many shapes it added.
fn fps_counter(ctx: &egui::Context, top: f64) -> usize {
    use egui::{Color32, CornerRadius, FontId, Id, LayerId, Order, Rect, pos2, vec2};
    let fps = f32::from_bits(crate::mugetsu::SHOWN_FPS.load(std::sync::atomic::Ordering::Relaxed));
    let text = format!("{fps:.0} FPS");
    let origin = pos2(8.0, top as f32 / ctx.pixels_per_point() + 8.0);
    let painter = ctx.layer_painter(LayerId::new(Order::Background, Id::new("mugetsu_fps")));
    let galley = painter.layout_no_wrap(text, FontId::monospace(12.0), Color32::from_rgb(232, 234, 242));
    let badge = Rect::from_min_size(origin, galley.size() + vec2(22.0, 8.0));
    let dot = if fps >= 50.0 {
        Color32::from_rgb(124, 220, 140)
    } else if fps >= 24.0 {
        Color32::from_rgb(232, 210, 120)
    } else {
        Color32::from_rgb(240, 120, 110)
    };
    painter.rect_filled(badge, CornerRadius::same(7), Color32::from_rgba_premultiplied(14, 16, 22, 200));
    painter.circle_filled(pos2(badge.left() + 9.0, badge.center().y), 3.0, dot);
    painter.galley(badge.left_top() + vec2(16.0, 4.0), galley, Color32::from_rgb(232, 234, 242));
    3
}

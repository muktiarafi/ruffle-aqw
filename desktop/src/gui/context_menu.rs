use crate::custom_event::RuffleEvent;
use crate::mugetsu::{FpsLimit, Graphics, MenuSettings, Performance, Setting};
use egui::{
    Align, Align2, Area, Button, Checkbox, Color32, CornerRadius, FontId, Frame, Id, Key,
    KeyboardShortcut, Layout, Margin, Modifiers, Order, Pos2, Rect, Response, Sense, Shadow,
    Stroke, Style, Ui, Widget, pos2, vec2,
};
use ruffle_core::{ContextMenuItem, PlayerEvent};
use unic_langid::LanguageIdentifier;
use winit::event_loop::EventLoopProxy;

use super::text;

pub struct ContextMenu {
    items: Vec<ContextMenuItem>,
    /// AQW build: Mugetsu's settings, below the movie's items (AQW has none of its own).
    settings: Option<MenuSettings>,
    position: Option<Pos2>,
    close_event: PlayerEvent,
}

impl ContextMenu {
    pub fn new(
        items: Vec<ContextMenuItem>,
        settings: Option<MenuSettings>,
        close_event: PlayerEvent,
    ) -> Self {
        Self {
            items,
            settings,
            position: None,
            close_event,
        }
    }

    pub fn close_event(self) -> PlayerEvent {
        self.close_event
    }

    pub fn show(
        &mut self,
        locale: &LanguageIdentifier,
        egui_ctx: &egui::Context,
        event_loop: &EventLoopProxy<RuffleEvent>,
        fullscreen: bool,
    ) -> bool {
        if let Some(settings) = self.settings {
            return self.show_mugetsu(egui_ctx, event_loop, settings);
        }
        let mut item_clicked = false;
        self.position = self.position.or(egui_ctx.pointer_latest_pos());

        let area = Area::new(Id::new("context_menu"))
            .order(Order::Foreground)
            .fixed_pos(self.position.unwrap_or_default())
            .constrain_to(egui_ctx.content_rect())
            .interactable(true)
            .show(egui_ctx, |ui| {
                set_menu_style(ui.style_mut());
                Frame::menu(ui.style()).show(ui, |ui| {
                    ui.set_max_width(150.0);
                    ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                        for (i, item) in self.items.iter().enumerate() {
                            if i != 0 && item.separator_before {
                                ui.separator();
                            }
                            let clicked = if item.checked {
                                Checkbox::new(&mut true, &item.caption).ui(ui).clicked()
                            } else {
                                let button = Button::new(&item.caption)
                                    .wrap_mode(egui::TextWrapMode::Extend);

                                ui.add_enabled(item.enabled, button).clicked()
                            };
                            if clicked {
                                let _ =
                                    event_loop.send_event(RuffleEvent::ContextMenuItemClicked(i));
                                item_clicked = true;
                            }
                        }

                        if fullscreen {
                            ui.separator();
                            if Button::new(text(locale, "context-menu-exit-fullscreen"))
                                .shortcut_text(ui.ctx().format_shortcut(&KeyboardShortcut::new(
                                    Modifiers::NONE,
                                    Key::Escape,
                                )))
                                .wrap_mode(egui::TextWrapMode::Extend)
                                .ui(ui)
                                .clicked()
                            {
                                let _ = event_loop.send_event(RuffleEvent::ExitFullScreen);
                                item_clicked = true;
                            }
                        }
                    })
                })
            });

        let should_close = item_clicked
            || area.response.clicked_elsewhere()
            || egui_ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Escape));

        !should_close
    }

    /// AQW build: Mugetsu's settings panel (the movie's own items, if any, on top).
    fn show_mugetsu(
        &mut self,
        egui_ctx: &egui::Context,
        event_loop: &EventLoopProxy<RuffleEvent>,
        settings: MenuSettings,
    ) -> bool {
        let mut item_clicked = false;
        if self.position.is_none() {
            // Just opened: every setting is one line again.
            Rows::close_all(egui_ctx, MENU_ID);
        }
        self.position = self.position.or(egui_ctx.pointer_latest_pos());
        // Sized with the window: as designed at about 1000 points tall (a maximized window on
        // a 1080p screen), smaller in a smaller window (down to 60%), so it never covers half of
        // a small game window. Everything is laid out at that size (text rendered sharp at it).
        let z = (egui_ctx.content_rect().height() / 1000.0).clamp(0.6, 1.0);
        let send = |event| {
            let _ = event_loop.send_event(event);
        };
        let mut changes = Vec::new();

        let area = Area::new(Id::new("context_menu"))
            .order(Order::Foreground)
            .fixed_pos(self.position.unwrap_or_default())
            .constrain_to(egui_ctx.content_rect())
            .interactable(true)
            .show(egui_ctx, |ui| {
                panel_style(ui.style_mut());
                Frame::new()
                    .fill(PANEL)
                    .stroke(Stroke::new(1.0, EDGE))
                    .corner_radius(CornerRadius::same((12.0 * z) as u8))
                    .inner_margin(Margin::same((10.0 * z) as i8))
                    .shadow(Shadow {
                        offset: [0, (6.0 * z) as i8],
                        blur: (24.0 * z) as u8,
                        spread: 0,
                        color: Color32::from_black_alpha(140),
                    })
                    .show(ui, |ui| {
                        ui.set_width(WIDTH * z);
                        ui.spacing_mut().item_spacing = vec2(0.0, 2.0 * z);

                        // Header.
                        let (rect, _) =
                            ui.allocate_exact_size(vec2(WIDTH * z, 24.0 * z), Sense::hover());
                        ui.painter().text(
                            pos2(rect.left() + 4.0 * z, rect.center().y),
                            Align2::LEFT_CENTER,
                            "Mugetsu",
                            FontId::proportional(16.0 * z),
                            TEXT,
                        );
                        ui.painter().text(
                            pos2(rect.right() - 4.0 * z, rect.center().y),
                            Align2::RIGHT_CENTER,
                            format!("v{}", crate::mugetsu::VERSION),
                            FontId::proportional(12.0 * z),
                            MUTED,
                        );
                        ui.add_space(4.0 * z);

                        // The movie's own items (AQW has none).
                        if !self.items.is_empty() {
                            for (i, item) in self.items.iter().enumerate() {
                                if !item.enabled {
                                    continue;
                                }
                                if action_row(ui, z, &item.caption, item.checked.then_some("✓"))
                                    .clicked()
                                {
                                    send(RuffleEvent::ContextMenuItemClicked(i));
                                    item_clicked = true;
                                }
                            }
                        }

                        // Mugetsu's settings (the same rows as the menu bar's "Mugetsu" menu).
                        if settings_rows(ui, z, &settings, MENU_ID, &mut changes) {
                            item_clicked = true;
                        }
                    })
            });
        for setting in changes {
            // The settings this menu shows are the ones it was opened with, and after
            // "Custom" it stays open for the number: it has to show that pick itself. (It did
            // not: the field for the number only appeared once the menu was opened again.)
            if let Some(shown) = &mut self.settings {
                match setting {
                    Setting::FpsLimit(limit) => shown.fps_limit = limit,
                    Setting::MemoryLimit(mb) => shown.memory_limit_mb = mb,
                    _ => {}
                }
            }
            send(RuffleEvent::MugetsuSetting(setting));
        }

        let should_close = item_clicked
            || area.response.clicked_elsewhere()
            || egui_ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Escape));

        !should_close
    }
}

// --- Mugetsu's panel -------------------------------------------------------------------------

pub(crate) const WIDTH: f32 = 268.0;
pub(crate) const PANEL: Color32 = Color32::from_rgba_premultiplied(17, 19, 26, 242);
pub(crate) const EDGE: Color32 = Color32::from_rgba_premultiplied(58, 62, 78, 255);
pub(crate) const ROW_HOVER: Color32 = Color32::from_rgba_premultiplied(40, 44, 58, 255);
pub(crate) const TRACK: Color32 = Color32::from_rgba_premultiplied(48, 52, 66, 255);
pub(crate) const ACCENT: Color32 = Color32::from_rgb(124, 108, 240);
pub(crate) const TEXT: Color32 = Color32::from_rgb(232, 234, 242);
pub(crate) const MUTED: Color32 = Color32::from_rgb(140, 146, 166);

/// The graphics API choices on this system, and the values they stand for.
pub(crate) fn graphics_options() -> (&'static [&'static str], &'static [Option<Graphics>]) {
    #[cfg(target_os = "linux")]
    return (
        &["Auto", "Vulkan", "OpenGL"],
        &[None, Some(Graphics::Vulkan), Some(Graphics::Gl)],
    );
    #[cfg(target_os = "macos")]
    return (&["Auto", "Metal"], &[None, Some(Graphics::Metal)]);
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return (
        &["Auto", "Vulkan", "DirectX 12"],
        &[None, Some(Graphics::Vulkan), Some(Graphics::Dx12)],
    );
}

/// Performance mode's choices, and the values they stand for.
pub(crate) fn performance_options() -> (&'static [&'static str], &'static [Performance]) {
    (
        &["Off", "Balanced", "Fast"],
        &[Performance::Off, Performance::Balanced, Performance::Fast],
    )
}

/// Mugetsu's settings as rows, the same in the right-click menu and in the menu bar's
/// "Mugetsu" menu (`id` keeps the two apart): one line per setting with its value on the
/// right. A setting with choices opens under its line when clicked, one at a time, and the
/// line at the bottom says what the row under the mouse does. (Before, every setting showed
/// its name, a hint and its choices all the time: twice as tall, and crowded.) Every change
/// made in this pass is pushed to `changes`; returns whether one of them settles things, which
/// closes the menu (a "Custom" choice does not: its number is typed next).
pub(crate) fn settings_rows(
    ui: &mut Ui,
    z: f32,
    settings: &MenuSettings,
    id: &str,
    changes: &mut Vec<Setting>,
) -> bool {
    let mut rows = Rows::new(ui, z, id);
    let mut done = false;

    // The picture. (Edge smoothing was a row here, Flash-style or MSAA, until 2.0.0: see
    // `main`.)
    let (options, values) = performance_options();
    let current = values
        .iter()
        .position(|&level| level == settings.performance)
        .unwrap_or(0);
    if let Some(choice) = rows.choice(
        ui,
        "Performance mode",
        "Fewer pixels: faster on weak graphics",
        options,
        current,
    ) {
        changes.push(Setting::Performance(values[choice]));
        done = true;
    }
    if settings.graphics_choice {
        let (options, values) = graphics_options();
        let current = values
            .iter()
            .position(|&graphics| graphics == settings.graphics)
            .unwrap_or(0);
        if let Some(choice) =
            rows.choice(ui, "Graphics API", "Changing it restarts Mugetsu", options, current)
        {
            changes.push(Setting::Graphics(values[choice]));
            done = true;
        }
    }

    // The frames.
    rows.divider(ui);
    if rows.toggle(
        ui,
        "Frame interpolation",
        "Smoother motion, best on strong PCs",
        settings.interpolation,
    ) {
        changes.push(Setting::Interpolation(!settings.interpolation));
        done = true;
    }
    if settings.interpolation {
        if let Some(choice) = rows.choice(
            ui,
            "Frame pacing",
            "Uncapped is for G-Sync / FreeSync",
            &["V-sync", "Uncapped"],
            if settings.uncapped { 1 } else { 0 },
        ) {
            changes.push(Setting::Uncapped(choice == 1));
            done = true;
        }
        if settings.uncapped {
            let value = match settings.fps_limit {
                FpsLimit::Off => "Off".to_owned(),
                FpsLimit::Screen => "Screen".to_owned(),
                FpsLimit::Fixed(fps) => fps.to_string(),
            };
            if rows.value(ui, "FPS limit", "Caps uncapped frames", &value)
                && let Some((limit, settled)) =
                    fps_limit_rows(ui, z, &format!("fps_limit_{id}"), settings.fps_limit)
            {
                changes.push(Setting::FpsLimit(limit));
                done |= settled;
            }
        }
    }

    // The memory.
    rows.divider(ui);
    let value = match settings.memory_limit_mb {
        0 => "Off".to_owned(),
        1024 => "1 GB".to_owned(),
        2048 => "2 GB".to_owned(),
        3072 => "3 GB".to_owned(),
        mb => format!("{mb} MB"),
    };
    if rows.value(ui, "Memory limit", "Mugetsu tries to stay under it", &value)
        && let Some((mb, settled)) =
            memory_limit_rows(ui, z, &format!("memory_{id}"), settings.memory_limit_mb)
    {
        changes.push(Setting::MemoryLimit(mb));
        done |= settled;
    }
    if rows.action(ui, "Free memory now", "Lets go of what the game is not using") {
        changes.push(Setting::FreeMemory);
        done = true;
    }

    // The rest.
    rows.divider(ui);
    if rows.toggle(ui, "Show FPS", "A small counter in the corner", settings.show_fps) {
        changes.push(Setting::ShowFps(!settings.show_fps));
        done = true;
    }
    if rows.toggle(
        ui,
        "Update reminders",
        "Tells you when a new version is out",
        settings.update_reminders,
    ) {
        changes.push(Setting::UpdateReminders(!settings.update_reminders));
        done = true;
    }
    rows.footer(ui);
    done
}

/// The right-click menu's `id` for `settings_rows`.
pub(crate) const MENU_ID: &str = "menu";

/// One pass over the settings' rows: which setting is open, and what the bottom line says.
struct Rows<'a> {
    z: f32,
    id: &'a str,
    /// The setting whose choices are shown under its line (kept in egui's memory from one
    /// pass to the next).
    open: Option<Id>,
    /// What the row under the mouse does, and what the open one does.
    hint: Option<&'static str>,
    open_hint: Option<&'static str>,
}

impl<'a> Rows<'a> {
    fn new(ui: &Ui, z: f32, id: &'a str) -> Self {
        // TEST SWITCH: `RUFFLE_TEST_MENU_OPEN=<a setting's name>` shows that setting open, for
        // a picture of it (a test movie cannot click).
        static TEST_OPEN: std::sync::OnceLock<Option<Id>> = std::sync::OnceLock::new();
        let test = *TEST_OPEN.get_or_init(|| {
            std::env::var("RUFFLE_TEST_MENU_OPEN")
                .ok()
                .map(|title| Id::new(title.as_str()))
        });
        let open = ui.data(|data| data.get_temp::<Id>(Self::open_key(id))).or(test);
        Self {
            z,
            id,
            open,
            hint: None,
            open_hint: None,
        }
    }

    fn open_key(id: &str) -> Id {
        Id::new(("mugetsu_open_row", id))
    }

    /// Every setting of the menu `id` is one line again.
    pub(crate) fn close_all(ctx: &egui::Context, id: &str) {
        ctx.data_mut(|data| data.remove::<Id>(Self::open_key(id)));
    }

    /// A setting's line: its name, its value and a small arrow. A click opens it (closing the
    /// one that was open) or closes it. Returns whether it is open: its controls come next.
    fn value(&mut self, ui: &mut Ui, title: &'static str, hint: &'static str, value: &str) -> bool {
        let z = self.z;
        let row = Id::new(title);
        let (rect, response) = ui.allocate_exact_size(vec2(WIDTH * z, ROW * z), Sense::click());
        if response.hovered() {
            ui.painter().rect_filled(rect, CornerRadius::same((8.0 * z) as u8), ROW_HOVER);
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            self.hint = Some(hint);
        }
        if response.clicked() {
            self.open = (self.open != Some(row)).then_some(row);
            let key = Self::open_key(self.id);
            ui.data_mut(|data| match self.open {
                Some(row) => {
                    data.insert_temp(key, row);
                }
                None => data.remove::<Id>(key),
            });
        }
        let open = self.open == Some(row);
        if open {
            self.open_hint = Some(hint);
        }
        let painter = ui.painter();
        painter.text(
            pos2(rect.left() + 8.0 * z, rect.center().y),
            Align2::LEFT_CENTER,
            title,
            FontId::proportional(14.0 * z),
            TEXT,
        );
        painter.text(
            pos2(rect.right() - 26.0 * z, rect.center().y),
            Align2::RIGHT_CENTER,
            value,
            FontId::proportional(13.0 * z),
            if open { TEXT } else { MUTED },
        );
        // The arrow: down while closed, up while open.
        let center = pos2(rect.right() - 13.0 * z, rect.center().y);
        let (half, tip) = (4.0 * z, if open { -2.0 * z } else { 2.0 * z });
        let stroke = Stroke::new(1.5 * z, MUTED);
        painter.line_segment([center + vec2(-half, -tip), center + vec2(0.0, tip)], stroke);
        painter.line_segment([center + vec2(0.0, tip), center + vec2(half, -tip)], stroke);
        open
    }

    /// A setting with a few choices: its line, and while it is open the choices under it.
    /// Returns the choice clicked, if it isn't `selected`.
    fn choice(
        &mut self,
        ui: &mut Ui,
        title: &'static str,
        hint: &'static str,
        options: &[&str],
        selected: usize,
    ) -> Option<usize> {
        if !self.value(ui, title, hint, options[selected]) {
            return None;
        }
        segmented(ui, self.z, &format!("{title}_{}", self.id), options, selected)
    }

    /// A setting that is on or off: its name and a switch. Returns whether it was clicked.
    fn toggle(&mut self, ui: &mut Ui, title: &'static str, hint: &'static str, on: bool) -> bool {
        let z = self.z;
        let (rect, response) = ui.allocate_exact_size(vec2(WIDTH * z, ROW * z), Sense::click());
        if response.hovered() {
            ui.painter().rect_filled(rect, CornerRadius::same((8.0 * z) as u8), ROW_HOVER);
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            self.hint = Some(hint);
        }
        ui.painter().text(
            pos2(rect.left() + 8.0 * z, rect.center().y),
            Align2::LEFT_CENTER,
            title,
            FontId::proportional(14.0 * z),
            TEXT,
        );
        // The switch, sliding (animated) between off and on.
        let how_on = ui.ctx().animate_bool_responsive(response.id, on);
        let switch = Rect::from_center_size(
            pos2(rect.right() - 24.0 * z, rect.center().y),
            vec2(32.0 * z, 18.0 * z),
        );
        let radius = switch.height() / 2.0;
        let track = lerp_color(TRACK, ACCENT, how_on);
        ui.painter().rect_filled(switch, CornerRadius::same(radius as u8), track);
        let knob_x = egui::lerp((switch.left() + radius)..=(switch.right() - radius), how_on);
        ui.painter()
            .circle_filled(pos2(knob_x, switch.center().y), radius - 3.0 * z, Color32::WHITE);
        response.clicked()
    }

    /// A row that does something at once. Returns whether it was clicked.
    fn action(&mut self, ui: &mut Ui, title: &'static str, hint: &'static str) -> bool {
        let response = action_row(ui, self.z, title, None);
        if response.hovered() {
            self.hint = Some(hint);
        }
        response.clicked()
    }

    /// A thin line between two groups of settings.
    fn divider(&self, ui: &mut Ui) {
        let z = self.z;
        let (rect, _) = ui.allocate_exact_size(vec2(WIDTH * z, 7.0 * z), Sense::hover());
        ui.painter().hline(
            (rect.left() + 6.0 * z)..=(rect.right() - 6.0 * z),
            rect.center().y,
            Stroke::new(1.0, TRACK),
        );
    }

    /// The bottom line: what the row under the mouse does, or else the open one.
    fn footer(self, ui: &mut Ui) {
        let z = self.z;
        self.divider(ui);
        let (rect, _) = ui.allocate_exact_size(vec2(WIDTH * z, 20.0 * z), Sense::hover());
        ui.painter().text(
            pos2(rect.left() + 8.0 * z, rect.center().y),
            Align2::LEFT_CENTER,
            self.hint
                .or(self.open_hint)
                .unwrap_or("Point at a setting to see what it does"),
            FontId::proportional(11.5 * z),
            MUTED,
        );
    }
}

/// The height of one setting's line, at full size.
const ROW: f32 = 30.0;

/// The keyboard goes to the number field of `id` (see `number_row`) when it is next drawn.
fn focus_number_row(ui: &Ui, id: &str) {
    ui.data_mut(|data| data.insert_temp(Id::new(("mugetsu_number_focus", id)), true));
}

/// The memory limit's rows: what the limit is compared with right now, then none, three
/// presets and "Custom", and under "Custom" a field to type the MB in (applied with Enter).
/// Returns the limit in MB picked or typed in this pass (0: none), and whether that is the end
/// of it (false for "Custom" itself).
///
/// The limit counts what Task Manager shows (see `mugetsu::task_manager_mb`; the user's
/// decision of 2026-11-16, with these presets): the line with the figure says the same
/// number. (While it counted committed memory the presets were 3, 4 and 6 GB.)
pub(crate) fn memory_limit_rows(ui: &mut Ui, z: f32, id: &str, current_mb: u32) -> Option<(u32, bool)> {
    const OPTIONS: [&str; 5] = ["Off", "1 GB", "2 GB", "3 GB", "Custom"];
    /// What "Custom" starts at, until a number is typed.
    const FIRST_CUSTOM_MB: u32 = 1500;
    info_row(
        ui,
        z,
        &format!(
            "Now: {:.1} GB",
            crate::mugetsu::memory_now_mb() as f32 / 1024.0
        ),
    );
    let selected = match current_mb {
        0 => 0,
        1024 => 1,
        2048 => 2,
        3072 => 3,
        _ => 4,
    };
    let mut picked = match segmented(ui, z, id, &OPTIONS, selected) {
        Some(0) => Some((0, true)),
        Some(1) => Some((1024, true)),
        Some(2) => Some((2048, true)),
        Some(3) => Some((3072, true)),
        Some(_) => {
            focus_number_row(ui, id);
            Some((FIRST_CUSTOM_MB, false))
        }
        None => None,
    };
    if selected == 4
        && let Some(typed) =
            number_row(ui, z, id, "Custom (MB)", "Enter to apply", current_mb, 512..=65536)
    {
        picked = Some((typed, true));
    }
    picked
}

/// A line of small gray text under an open setting's name.
fn info_row(ui: &mut Ui, z: f32, text: &str) {
    let (rect, _) = ui.allocate_exact_size(vec2(WIDTH * z, 18.0 * z), Sense::hover());
    ui.painter().text(
        pos2(rect.left() + 8.0 * z, rect.center().y),
        Align2::LEFT_CENTER,
        text,
        FontId::proportional(11.5 * z),
        MUTED,
    );
}

/// The FPS limit's rows while uncapped: the presets and "Custom", and under "Custom" a field
/// to type the number in (applied with Enter). Returns the limit picked or typed in this
/// pass, and whether that is the end of it (false for "Custom" itself: its number comes next).
pub(crate) fn fps_limit_rows(
    ui: &mut Ui,
    z: f32,
    id: &str,
    current: FpsLimit,
) -> Option<(FpsLimit, bool)> {
    const OPTIONS: [&str; 5] = ["Off", "Screen", "120", "60", "Custom"];
    /// What "Custom" starts at, until a number is typed.
    const FIRST_CUSTOM: u32 = 144;
    let selected = match current {
        FpsLimit::Off => 0,
        FpsLimit::Screen => 1,
        FpsLimit::Fixed(120) => 2,
        FpsLimit::Fixed(60) => 3,
        FpsLimit::Fixed(_) => 4,
    };
    info_row(ui, z, "Screen: just under your monitor's rate");
    // TEST SWITCH: `RUFFLE_TEST_MENU_FOCUS=1` gives the keyboard to the "Custom" field once, as
    // a click on "Custom" does (a test movie cannot click): for what an open field costs.
    static TEST_FOCUS: std::sync::Once = std::sync::Once::new();
    TEST_FOCUS.call_once(|| {
        if std::env::var_os("RUFFLE_TEST_MENU_FOCUS").is_some() {
            focus_number_row(ui, id);
        }
    });
    let mut picked = match segmented(ui, z, id, &OPTIONS, selected) {
        Some(0) => Some((FpsLimit::Off, true)),
        Some(1) => Some((FpsLimit::Screen, true)),
        Some(2) => Some((FpsLimit::Fixed(120), true)),
        Some(3) => Some((FpsLimit::Fixed(60), true)),
        Some(_) => {
            focus_number_row(ui, id);
            Some((FpsLimit::Fixed(FIRST_CUSTOM), false))
        }
        None => None,
    };
    if let (4, FpsLimit::Fixed(fps)) = (selected, current)
        && let Some(typed) = number_row(ui, z, id, "Custom limit", "Enter to apply", fps, 45..=1000)
    {
        picked = Some((FpsLimit::Fixed(typed), true));
    }
    picked
}

/// A row with a title and a field to type a whole number in. Returns the number once Enter
/// is pressed on one within `range`; until then the field shows `current`, or what is being
/// typed while it has the keyboard (kept in egui's memory under `id`). After
/// `focus_number_row` the field takes the keyboard and starts empty, ready for the number.
pub(crate) fn number_row(
    ui: &mut Ui,
    z: f32,
    id: &str,
    title: &str,
    hint: &str,
    current: u32,
    range: std::ops::RangeInclusive<u32>,
) -> Option<u32> {
    let (rect, _) = ui.allocate_exact_size(vec2(WIDTH * z, 30.0 * z), Sense::hover());
    ui.painter().text(
        pos2(rect.left() + 8.0 * z, rect.center().y),
        Align2::LEFT_CENTER,
        title,
        FontId::proportional(13.5 * z),
        TEXT,
    );
    let field = Rect::from_min_size(
        pos2(rect.right() - 74.0 * z, rect.top() + 3.0 * z),
        vec2(66.0 * z, 24.0 * z),
    );
    ui.painter().text(
        pos2(field.left() - 8.0 * z, rect.center().y),
        Align2::RIGHT_CENTER,
        hint,
        FontId::proportional(11.5 * z),
        MUTED,
    );
    let key = Id::new(("mugetsu_number", id));
    let mut text = ui
        .data(|data| data.get_temp::<String>(key))
        .unwrap_or_else(|| current.to_string());
    let response = ui.put(
        field,
        egui::TextEdit::singleline(&mut text)
            .char_limit(5)
            .font(FontId::proportional(13.0 * z))
            .horizontal_align(Align::Center),
    );
    let focus_key = Id::new(("mugetsu_number_focus", id));
    if ui.data(|data| data.get_temp::<bool>(focus_key)).is_some() {
        ui.data_mut(|data| {
            data.remove::<bool>(focus_key);
            data.insert_temp(key, String::new());
        });
        response.request_focus();
        return None;
    }
    text.retain(|c| c.is_ascii_digit());
    let entered = response.lost_focus() && ui.input(|input| input.key_pressed(Key::Enter));
    let typed = text.parse::<u32>().ok().filter(|number| range.contains(number));
    if response.has_focus() {
        ui.data_mut(|data| data.insert_temp(key, text));
    } else {
        // Without the keyboard the field shows the number in use again.
        ui.data_mut(|data| data.remove::<String>(key));
    }
    if entered { typed } else { None }
}

pub(crate) fn panel_style(style: &mut Style) {
    style.visuals.override_text_color = Some(TEXT);
    style.spacing.button_padding = vec2(0.0, 0.0);
    style.interaction.selectable_labels = false;
}

/// Pill-shaped choices side by side; returns the one clicked, if it isn't `selected`.
pub(crate) fn segmented(ui: &mut Ui, z: f32, id: &str, options: &[&str], selected: usize) -> Option<usize> {
    let (rect, _) = ui.allocate_exact_size(vec2(WIDTH * z, 30.0 * z), Sense::hover());
    let rect = rect.shrink2(vec2(6.0 * z, 2.0 * z));
    ui.painter().rect_filled(rect, CornerRadius::same((13.0 * z) as u8), TRACK);
    // Each choice gets room by the length of its word: of equal widths, "Custom" touched the
    // menu's edge beside "Off".
    let weights: Vec<f32> = options.iter().map(|option| option.chars().count() as f32 + 3.0).collect();
    let total: f32 = weights.iter().sum();
    let mut left = rect.left();
    let mut chosen = None;
    for (i, option) in options.iter().enumerate() {
        let width = rect.width() * weights[i] / total;
        let cell = Rect::from_min_size(pos2(left, rect.top()), vec2(width, rect.height()));
        left += width;
        let response = ui.interact(cell, Id::new((id, i)), Sense::click());
        let active = ui.ctx().animate_bool_responsive(response.id, i == selected);
        let pill = cell.shrink(2.0 * z);
        let pill_radius = CornerRadius::same((11.0 * z) as u8);
        if active > 0.0 {
            ui.painter().rect_filled(pill, pill_radius, ACCENT.gamma_multiply(active));
        } else if response.hovered() {
            ui.painter().rect_filled(pill, pill_radius, ROW_HOVER);
        }
        ui.painter().text(
            cell.center(),
            Align2::CENTER_CENTER,
            *option,
            FontId::proportional(13.0 * z),
            if i == selected { Color32::WHITE } else { MUTED },
        );
        if response.hovered() && i != selected {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if response.clicked() && i != selected {
            chosen = Some(i);
        }
    }
    chosen
}

/// A clickable row with a title (and something on the right, like an arrow).
pub(crate) fn action_row(ui: &mut Ui, z: f32, title: &str, right: Option<&str>) -> Response {
    let (rect, response) = ui.allocate_exact_size(vec2(WIDTH * z, 30.0 * z), Sense::click());
    if response.hovered() {
        ui.painter().rect_filled(rect, CornerRadius::same((8.0 * z) as u8), ROW_HOVER);
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    ui.painter().text(
        pos2(rect.left() + 8.0 * z, rect.center().y),
        Align2::LEFT_CENTER,
        title,
        FontId::proportional(14.5 * z),
        TEXT,
    );
    if let Some(right) = right {
        ui.painter().text(
            pos2(rect.right() - 10.0 * z, rect.center().y),
            Align2::RIGHT_CENTER,
            right,
            FontId::proportional(13.0 * z),
            MUTED,
        );
    }
    response
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_premultiplied(
        mix(a.r(), b.r()),
        mix(a.g(), b.g()),
        mix(a.b(), b.b()),
        mix(a.a(), b.a()),
    )
}

// Shamelessly stolen from egui menu::set_menu_style, a private internal function
fn set_menu_style(style: &mut Style) {
    style.spacing.button_padding = vec2(2.0, 0.0);
    style.visuals.widgets.active.bg_stroke = Stroke::NONE;
    style.visuals.widgets.hovered.bg_stroke = Stroke::NONE;
    style.visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
    style.visuals.widgets.inactive.bg_stroke = Stroke::NONE;
}

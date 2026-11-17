//! AQW build (Mugetsu): questions and notices in the game window, in the style of Mugetsu's
//! right-click menu: before restarting with another graphics API, a new version, and the
//! one-time graphics tips. Native message boxes were unreliable: one opened from winit's event
//! handler was never shown (see `gui::RuffleGui::ask_graphics_restart`), and those from
//! background threads could open behind the game.

use crate::gui::context_menu::{ACCENT, EDGE, MUTED, PANEL, ROW_HOVER, TEXT, TRACK, panel_style};
use crate::mugetsu::{Graphics, Notice};
use egui::{
    Align2, Color32, CornerRadius, FontId, Frame, Id, Key, Label, Layout, Margin, Modal,
    Modifiers, Response, RichText, Sense, Shadow, Stroke, Ui, vec2,
};

const WIDTH: f32 = 330.0;

/// What answering a prompt does.
pub enum Action {
    RestartWithGraphics(Option<Graphics>),
    OpenReleasesPage,
    OpenGraphicsSettings,
    /// No more reminders for this version (newer ones still are).
    SkipVersion(String),
    /// No more reminders at all (the right-click menu turns them back on).
    StopUpdateReminders,
    /// Mugetsu starts again as it is (see `Prompt::interpolation_restart`).
    Restart,
    /// Performance mode at this level (see `Notice::PerformanceMode`).
    TurnOnPerformance(crate::mugetsu::Performance),
}

/// How a prompt was answered: a button or a link (by index), or closed without either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Picked {
    Button(usize),
    Link(usize),
    Dismissed,
}

pub struct Prompt {
    title: String,
    body: String,
    hint: Option<&'static str>,
    /// Right to left; the first is the main one (also Enter). `None` just closes the prompt.
    buttons: Vec<(&'static str, Option<Action>)>,
    /// Smaller choices below the buttons.
    links: Vec<(&'static str, Action)>,
    /// The player asked for this (picked something in a menu), so Enter picks the main button
    /// and Esc or a click outside closes it. A notice, which appears by itself, needs one of
    /// its buttons: the player may be typing or clicking in the game right then (keys still
    /// reach the game while a prompt is shown), and Enter in the chat opened the download page.
    asked: bool,
}

impl Prompt {
    pub fn graphics(choice: Option<Graphics>) -> Self {
        let name = choice.map_or("automatic", Graphics::name);
        Self {
            title: "Restart Mugetsu?".to_string(),
            body: format!("Mugetsu will restart to use {name} graphics."),
            hint: Some(if cfg!(windows) {
                "You can switch again any time: right-click the game or the title bar."
            } else {
                "You can switch again any time: right-click the game."
            }),
            buttons: vec![
                ("Restart", Some(Action::RestartWithGraphics(choice))),
                ("Cancel", None),
            ],
            links: Vec::new(),
            asked: true,
        }
    }

    /// Shown when frame interpolation is turned on in a Mugetsu that started without it:
    /// the render thread, which is where interpolation gets its pictures from in a crowded
    /// room, is only set up when Mugetsu starts (see `mugetsu::render_thread_needs_restart`).
    /// It needs one of its buttons (Enter may be meant for the chat, and would restart).
    pub fn interpolation_restart() -> Self {
        Self {
            title: "Restart Mugetsu for full speed?".to_string(),
            body: "Frame interpolation is on. In crowded rooms it gets much smoother after a \
                   restart: Mugetsu then does its drawing on a second processor thread."
                .to_string(),
            hint: Some(
                "It works without a restart too, only slower. Restarting means logging in \
                 again.",
            ),
            buttons: vec![("Restart", Some(Action::Restart)), ("Later", None)],
            links: Vec::new(),
            asked: false,
        }
    }

    pub fn is_interpolation_restart(&self) -> bool {
        self.buttons
            .iter()
            .any(|(_, action)| matches!(action, Some(Action::Restart)))
    }

    pub fn notice(notice: Notice) -> Self {
        match notice {
            // Closing it (or "Remind me later") asks again at the next start, as does
            // "Download" while this version is still the one running.
            Notice::UpdateAvailable { latest, current } => Self {
                title: format!("Mugetsu {latest} is available"),
                body: format!("You have {current}. Open the download page?"),
                hint: None,
                buttons: vec![
                    ("Download", Some(Action::OpenReleasesPage)),
                    ("Remind me later", None),
                ],
                links: vec![
                    ("Skip this version", Action::SkipVersion(latest)),
                    ("Don't remind me again", Action::StopUpdateReminders),
                ],
                asked: false,
            },
            Notice::BuiltInGraphics { chosen, dedicated } => Self {
                title: "Mugetsu is on your built-in graphics".to_string(),
                body: if cfg!(windows) {
                    format!(
                        "It's running on {chosen} instead of your {dedicated}, which is much \
                        faster. To switch: in Windows' graphics settings, add Mugetsu.exe \
                        (Browse), click it > Options > High performance > Save. On NVIDIA \
                        laptops also check NVIDIA Control Panel > Manage 3D settings > Program \
                        settings. Then restart Mugetsu."
                    )
                } else {
                    format!(
                        "It's running on {chosen} instead of your {dedicated}, which is much \
                        faster. Set Mugetsu to use the dedicated graphics card in your system's \
                        per-app graphics setting, then restart Mugetsu."
                    )
                },
                hint: Some("This is only shown once."),
                buttons: if cfg!(windows) {
                    vec![
                        ("Open graphics settings", Some(Action::OpenGraphicsSettings)),
                        ("OK", None),
                    ]
                } else {
                    vec![("OK", None)]
                },
                links: Vec::new(),
                asked: false,
            },
            Notice::QualityLow { chip, quality } => Self {
                title: "Set Quality to Low for more FPS".to_string(),
                body: format!(
                    "AQW is running on your built-in graphics ({chip}) with Quality set to \
                    {quality}, which is slow on built-in graphics. Open AQW's Options and \
                    change Quality to Low: edges are a little less smooth, but built-in \
                    graphics usually run AQW several times faster."
                ),
                hint: Some("Mugetsu won't show this again."),
                buttons: vec![("OK", None)],
                links: Vec::new(),
                asked: false,
            },
            Notice::PerformanceMode { chip, level } => Self {
                title: "Turn on Performance mode?".to_string(),
                body: format!(
                    "AQW is running slower than it should on your built-in graphics ({chip}). \
                    Performance mode draws the game with fewer pixels and stretches it over \
                    the window: the picture gets a little softer, and the game runs faster."
                ),
                hint: Some(
                    "You can change it any time in the right-click menu. Mugetsu won't ask again.",
                ),
                buttons: vec![
                    ("Turn on", Some(Action::TurnOnPerformance(level))),
                    ("Not now", None),
                ],
                links: Vec::new(),
                asked: false,
            },
            Notice::VulkanUnavailable { name, slower } => Self {
                title: "Vulkan isn't available".to_string(),
                body: format!(
                    "Mugetsu is using {name} instead. {} Updating your graphics driver \
                    usually makes Vulkan available (NVIDIA GTX 600, AMD Radeon HD 7000, \
                    Intel 6th-gen Core or newer).",
                    if slower {
                        "It works, but it's much slower than Vulkan: lower FPS in crowded rooms."
                    } else {
                        "It runs about as well as Vulkan."
                    }
                ),
                hint: Some("This is only shown once."),
                buttons: vec![("OK", None)],
                links: Vec::new(),
                asked: false,
            },
        }
    }

    /// Whether this asks about a graphics API (a newer pick replaces it).
    pub fn is_graphics(&self) -> bool {
        self.buttons
            .iter()
            .any(|(_, action)| matches!(action, Some(Action::RestartWithGraphics(_))))
    }

    /// What the answer `picked` does, if anything.
    pub fn into_action(self, picked: Picked) -> Option<Action> {
        match picked {
            Picked::Button(i) => self.buttons.into_iter().nth(i).and_then(|(_, action)| action),
            Picked::Link(i) => self.links.into_iter().nth(i).map(|(_, action)| action),
            Picked::Dismissed => None,
        }
    }

    /// Shows the prompt; returns the answer once there is one. For one the player asked for
    /// (see `asked`), Enter picks the main button and Esc or a click outside closes it.
    pub fn show(&self, egui_ctx: &egui::Context) -> Option<Picked> {
        // Sized with the window, like the right-click menu.
        let z = (egui_ctx.content_rect().height() / 1000.0).clamp(0.6, 1.0);
        let mut picked = None;
        let modal = Modal::new(Id::new("mugetsu_prompt"))
            .backdrop_color(Color32::from_black_alpha(120))
            .frame(
                Frame::new()
                    .fill(PANEL)
                    .stroke(Stroke::new(1.0, EDGE))
                    .corner_radius(CornerRadius::same((12.0 * z) as u8))
                    .inner_margin(Margin::same((16.0 * z) as i8))
                    .shadow(Shadow {
                        offset: [0, (6.0 * z) as i8],
                        blur: (24.0 * z) as u8,
                        spread: 0,
                        color: Color32::from_black_alpha(140),
                    }),
            )
            .show(egui_ctx, |ui| {
                panel_style(ui.style_mut());
                ui.set_width(WIDTH * z);
                ui.spacing_mut().item_spacing = vec2(0.0, 6.0 * z);
                ui.label(RichText::new(&self.title).size(17.0 * z).color(TEXT));
                ui.add(Label::new(RichText::new(&self.body).size(14.0 * z).color(TEXT)).wrap());
                if let Some(hint) = self.hint {
                    ui.add(Label::new(RichText::new(hint).size(12.0 * z).color(MUTED)).wrap());
                }
                ui.add_space(8.0 * z);
                // As tall as the buttons: a right-to-left layout centered across takes all the
                // height left, which made the box taller than the window (text cut off).
                let row = vec2(WIDTH * z, 32.0 * z);
                ui.allocate_ui_with_layout(row, Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing = vec2(8.0 * z, 0.0);
                    for (i, (label, _)) in self.buttons.iter().enumerate() {
                        let clicked = if i == 0 {
                            button(ui, z, label, ACCENT, ACCENT.gamma_multiply(1.15))
                        } else {
                            button(ui, z, label, TRACK, ROW_HOVER)
                        }
                        .clicked();
                        if clicked {
                            picked = Some(Picked::Button(i));
                        }
                    }
                });
                if !self.links.is_empty() {
                    ui.add_space(4.0 * z);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = vec2(14.0 * z, 0.0);
                        for (i, (label, _)) in self.links.iter().enumerate() {
                            if link(ui, z, label).clicked() {
                                picked = Some(Picked::Link(i));
                            }
                        }
                    });
                }
            });
        if self.asked {
            if picked.is_none() && modal.should_close() {
                picked = Some(Picked::Dismissed);
            }
            if picked.is_none()
                && egui_ctx.input_mut(|input| input.consume_key(Modifiers::NONE, Key::Enter))
            {
                picked = Some(Picked::Button(0));
            }
        }
        picked
    }
}

/// A rounded button in the panel's style, at least 96 points wide.
fn button(ui: &mut Ui, z: f32, label: &str, fill: Color32, hover: Color32) -> Response {
    let font = FontId::proportional(14.0 * z);
    let text_width = ui.painter().layout_no_wrap(label.to_string(), font.clone(), TEXT).size().x;
    let size = vec2((text_width + 28.0 * z).max(96.0 * z), 32.0 * z);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let fill = if response.hovered() { hover } else { fill };
    ui.painter().rect_filled(rect, CornerRadius::same((10.0 * z) as u8), fill);
    ui.painter().text(rect.center(), Align2::CENTER_CENTER, label, font, Color32::WHITE);
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

/// Small gray text that brightens when hovered, for the less common choices.
fn link(ui: &mut Ui, z: f32, label: &str) -> Response {
    let font = FontId::proportional(12.0 * z);
    let galley = ui.painter().layout_no_wrap(label.to_string(), font.clone(), MUTED);
    let (rect, response) = ui.allocate_exact_size(galley.size(), Sense::click());
    let color = if response.hovered() { TEXT } else { MUTED };
    ui.painter().text(rect.left_center(), Align2::LEFT_CENTER, label, font, color);
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, PointerButton, RawInput, Rect, pos2};

    /// Window sizes in points: Mugetsu's own, a small one and a big one.
    const WINDOWS: [(f32, f32); 3] = [(960.0, 580.0), (640.0, 400.0), (1920.0, 1080.0)];

    /// Every kind of prompt, the graphics question first, with long names.
    fn prompts() -> Vec<Prompt> {
        vec![
            Prompt::graphics(None),
            Prompt::notice(Notice::UpdateAvailable {
                latest: "1.10.12".to_string(),
                current: "1.6.0".to_string(),
            }),
            Prompt::notice(Notice::BuiltInGraphics {
                chosen: "Intel(R) Iris(R) Xe Graphics".to_string(),
                dedicated: "NVIDIA GeForce RTX 4060 Laptop GPU".to_string(),
            }),
            Prompt::notice(Notice::QualityLow {
                chip: "Intel(R) UHD Graphics 620".to_string(),
                quality: "Medium",
            }),
            Prompt::notice(Notice::VulkanUnavailable {
                name: "OpenGL",
                slower: true,
            }),
            Prompt::notice(Notice::PerformanceMode {
                chip: "Intel(R) UHD Graphics 620".to_string(),
                level: crate::mugetsu::Performance::Fast,
            }),
            Prompt::interpolation_restart(),
        ]
    }

    /// Draws `prompt` once in a window of `size` with `events`; returns its answer.
    fn pass(
        ctx: &egui::Context,
        prompt: &Prompt,
        size: (f32, f32),
        events: Vec<Event>,
    ) -> Option<Picked> {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(size.0, size.1))),
            events,
            ..Default::default()
        };
        let mut picked = None;
        let mut output = ctx.run_ui(input, |ui| picked = prompt.show(ui.ctx()));
        // (Nothing draws it here; epaint checks that texture changes were not dropped unseen.)
        output.textures_delta.clear();
        picked
    }

    /// A context in which `prompt` has been drawn for a second of passes, and its box by
    /// then: a box that grows a little with each pass (see `prompts_fit_in_the_window`) no
    /// longer fits in the window.
    fn settled(prompt: &Prompt, size: (f32, f32)) -> (egui::Context, Rect) {
        let ctx = egui::Context::default();
        for _ in 0..60 {
            assert_eq!(pass(&ctx, prompt, size, Vec::new()), None);
        }
        let rect = ctx
            .memory(|memory| memory.area_rect(Id::new("mugetsu_prompt")))
            .expect("the prompt has a box");
        (ctx, rect)
    }

    /// The first answer `steps` (the events of one pass each) get from a settled prompt.
    fn answer(ctx: &egui::Context, prompt: &Prompt, steps: Vec<Vec<Event>>) -> Option<Picked> {
        steps
            .into_iter()
            .chain([Vec::new()])
            .find_map(|events| pass(ctx, prompt, WINDOWS[0], events))
    }

    fn key(key: Key) -> Vec<Vec<Event>> {
        vec![vec![Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        }]]
    }

    fn click(pos: egui::Pos2) -> Vec<Vec<Event>> {
        let button = |pressed| Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed,
            modifiers: Modifiers::NONE,
        };
        vec![vec![Event::PointerMoved(pos)], vec![button(true)], vec![button(false)]]
    }

    #[test]
    fn prompts_fit_in_the_window() {
        // The buttons' row once took all the height left. With links below it, the box then
        // grew with every pass until it was as tall as the window, its title and text cut off.
        for size in WINDOWS {
            let z = (size.1 / 1000.0).clamp(0.6, 1.0);
            let window = Rect::from_min_size(pos2(0.0, 0.0), vec2(size.0, size.1));
            for prompt in prompts() {
                let (_, rect) = settled(&prompt, size);
                let title = &prompt.title;
                assert!(window.contains_rect(rect), "{title:?} in {size:?}: {rect:?}");
                assert!(rect.height() < 0.9 * size.1, "{title:?} in {size:?}: {rect:?}");
                assert!(
                    rect.width() <= (WIDTH + 2.0 * 16.0) * z + 4.0,
                    "{title:?} in {size:?}: {rect:?}"
                );
            }
        }
    }

    #[test]
    fn only_a_prompt_the_player_asked_for_takes_enter_esc_and_clicks_outside() {
        let outside = pos2(5.0, 5.0);
        let asked = Prompt::graphics(None);
        for (events, picked) in [
            (key(Key::Enter), Picked::Button(0)),
            (key(Key::Escape), Picked::Dismissed),
            (click(outside), Picked::Dismissed),
        ] {
            let (ctx, _) = settled(&asked, WINDOWS[0]);
            assert_eq!(answer(&ctx, &asked, events), Some(picked));
        }
        for notice in prompts().into_iter().skip(1) {
            for events in [key(Key::Enter), key(Key::Escape), click(outside)] {
                let (ctx, _) = settled(&notice, WINDOWS[0]);
                assert_eq!(answer(&ctx, &notice, events), None, "{:?}", notice.title);
            }
        }
    }

    #[test]
    fn the_main_button_answers_and_keys_still_reach_the_game() {
        let z = (WINDOWS[0].1 / 1000.0).clamp(0.6, 1.0);
        // (Without links below the buttons, so the main button is at the bottom right.)
        for prompt in prompts().into_iter().filter(|prompt| prompt.links.is_empty()) {
            let (ctx, rect) = settled(&prompt, WINDOWS[0]);
            // Why notices take no keys: the game still gets them while a prompt is shown.
            assert!(!ctx.egui_wants_keyboard_input());
            let main_button = rect.right_bottom() - vec2((16.0 + 40.0) * z, (16.0 + 16.0) * z);
            assert_eq!(
                answer(&ctx, &prompt, click(main_button)),
                Some(Picked::Button(0)),
                "{:?}",
                prompt.title
            );
        }
    }
}

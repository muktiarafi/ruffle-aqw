//! AQW build (Mugetsu): the right-click menu's settings as a "Mugetsu" menu in Ruffle's menu
//! bar (shown with --gui). Changes go through the same events as the right-click menu, so
//! they're applied the same way.

use crate::custom_event::RuffleEvent;
use crate::gui::context_menu::*;
use crate::mugetsu::{MenuSettings, VERSION};
use egui::*;
use winit::event_loop::EventLoopProxy;

pub fn contents(ui: &mut Ui, event_loop: &EventLoopProxy<RuffleEvent>) {
    let send = |setting| {
        let _ = event_loop.send_event(RuffleEvent::MugetsuSetting(setting));
    };
    let settings = MenuSettings::now();
    panel_style(ui.style_mut());
    let z = 1.0;
    ui.set_width(WIDTH * z);
    ui.spacing_mut().item_spacing = vec2(0.0, 2.0 * z);

    // Header.
    let (rect, _) = ui.allocate_exact_size(vec2(WIDTH * z, 24.0 * z), Sense::hover());
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
        format!("v{VERSION}"),
        FontId::proportional(12.0 * z),
        MUTED,
    );
    ui.add_space(4.0 * z);

    let mut changes = Vec::new();
    let done = settings_rows(ui, z, &settings, "bar", &mut changes);
    for setting in changes {
        send(setting);
    }
    if done {
        ui.close();
    }
}

pub fn menu_button(ui: &mut Ui, event_loop: &EventLoopProxy<RuffleEvent>) {
    // A click on a setting's line opens its choices: the menu must stay (egui closes a menu
    // on any click inside it unless told otherwise). `contents` closes it after a change.
    let config = egui::containers::menu::MenuConfig::new()
        .close_behavior(egui::containers::PopupCloseBehavior::CloseOnClickOutside);
    egui::containers::menu::MenuButton::new("Mugetsu")
        .config(config)
        .ui(ui, |ui| contents(ui, event_loop));
}

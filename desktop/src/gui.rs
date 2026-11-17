pub(crate) mod context_menu;
mod controller;
pub mod dialogs;
mod locale;
mod menu_bar;
mod mugetsu_menu;
mod movie;
mod picker;
mod prompt;
mod theme;
mod widgets;

pub use controller::GuiController;
pub use dialogs::DialogDescriptor;
pub use locale::LocalizableText;
pub use locale::available_languages;
pub use locale::optional_text;
pub use locale::text;
pub use locale::text_with_args;
pub use movie::MovieView;
pub use picker::FilePicker;
use ruffle_frontend_utils::content::ContentDescriptor;
pub use theme::ThemePreference;

use crate::custom_event::RuffleEvent;
use crate::gui::context_menu::ContextMenu;
use crate::gui::prompt::{Action, Prompt};
use crate::player::LaunchOptions;
use crate::preferences::GlobalPreferences;
use dialogs::Dialogs;
use egui::*;
use menu_bar::MenuBar;
use rfd::AsyncFileDialog;
use ruffle_core::debug_ui::Message as DebugMessage;
use ruffle_core::{Player, PlayerEvent};
use std::sync::{MutexGuard, Weak};
use std::{fs, mem};
use winit::event_loop::EventLoopProxy;

/// Size of the top menu bar in pixels.
/// This is the offset at which the movie will be shown,
/// and added to the window size if trying to match a movie.
pub const MENU_HEIGHT: u32 = 24;

/// The main controller for the Ruffle GUI.
pub struct RuffleGui {
    event_loop: EventLoopProxy<RuffleEvent>,
    context_menu: Option<ContextMenu>,
    /// AQW build: Mugetsu's questions in the game window (see `prompt`), the first one shown.
    prompts: Vec<Prompt>,
    dialogs: Dialogs,
    menu_bar: MenuBar,

    was_suspended_before_debug: bool,
    preferences: GlobalPreferences,
}

impl RuffleGui {
    fn new(
        window: Weak<winit::window::Window>,
        event_loop: EventLoopProxy<RuffleEvent>,
        default_content: Option<ContentDescriptor>,
        default_launch_options: LaunchOptions,
        preferences: GlobalPreferences,
    ) -> Self {
        Self {
            was_suspended_before_debug: false,

            context_menu: None,
            prompts: Vec::new(),
            dialogs: Dialogs::new(
                preferences.clone(),
                default_launch_options.clone(),
                default_content,
                window,
                event_loop.clone(),
            ),
            menu_bar: MenuBar::new(
                event_loop.clone(),
                default_launch_options,
                preferences.clone(),
            ),

            event_loop,
            preferences,
        }
    }

    /// Renders all of the main Ruffle UI, including the main menu and context menus.
    fn update(
        &mut self,
        egui_ctx: &egui::Context,
        egui_ui: &mut egui::Ui,
        show_menu: bool,
        mut player: Option<&mut Player>,
        menu_height_offset: f64,
    ) {
        let locale = self.preferences.language();

        self.menu_bar
            .consume_shortcuts(egui_ctx, &mut self.dialogs, player.as_deref_mut());
        if show_menu {
            self.menu_bar
                .show(&locale, egui_ui, &mut self.dialogs, player.as_deref_mut());
        }

        self.dialogs.show(&locale, egui_ctx, player.as_deref_mut());

        if let Some(player) = player {
            let was_suspended = player.debug_ui().should_suspend_player();
            player.show_debug_ui(egui_ctx, menu_height_offset);
            if was_suspended != player.debug_ui().should_suspend_player() {
                if player.debug_ui().should_suspend_player() {
                    self.was_suspended_before_debug = !player.is_playing();
                    player.set_is_playing(false);
                } else {
                    player.set_is_playing(!self.was_suspended_before_debug);
                }
            }
            for item in player.debug_ui().items_to_save() {
                let dialog = AsyncFileDialog::new().set_file_name(&item.suggested_name);
                let picker = self.dialogs.file_picker();
                let result = picker.show_dialog(dialog, |d| d.save_file());
                if let Some(result) = result {
                    tokio::spawn(async move {
                        let Some(handle) = result.await else {
                            return;
                        };
                        let path = handle.path();
                        if let Err(e) = fs::write(path, item.data) {
                            tracing::error!(
                                "Couldn't save {} to {path:?}: {e}",
                                item.suggested_name,
                            );
                        }
                    });
                }
            }

            if let Some(context_menu) = &mut self.context_menu
                && !context_menu.show(&locale, egui_ctx, &self.event_loop, player.is_fullscreen())
            {
                self.close_context_menu(player);
            }
        };

        if let Some(prompt) = self.prompts.first()
            && let Some(picked) = prompt.show(egui_ctx)
        {
            match self.prompts.remove(0).into_action(picked) {
                Some(Action::RestartWithGraphics(choice)) => {
                    let _ = self.event_loop.send_event(RuffleEvent::MugetsuSetting(
                        crate::mugetsu::Setting::RestartWithGraphics(choice),
                    ));
                }
                Some(Action::OpenReleasesPage) => crate::mugetsu::open_releases_page(),
                Some(Action::OpenGraphicsSettings) => crate::mugetsu::open_graphics_settings(),
                Some(Action::SkipVersion(version)) => crate::mugetsu::skip_update(&version),
                Some(Action::StopUpdateReminders) => crate::mugetsu::set_update_reminders(false),
                Some(Action::Restart) => {
                    let _ = self
                        .event_loop
                        .send_event(RuffleEvent::MugetsuSetting(crate::mugetsu::Setting::Restart));
                }
                Some(Action::TurnOnPerformance(level)) => {
                    let _ = self.event_loop.send_event(RuffleEvent::MugetsuSetting(
                        crate::mugetsu::Setting::Performance(level),
                    ));
                }
                None => {}
            }
        }
    }

    /// AQW build: asks in the game window before restarting with `choice`. A Windows message
    /// box opened from winit's event handler was never shown: Windows shows one when its message
    /// queue goes idle, and inside the handler winit answered every repaint with another. The
    /// game looked frozen (its window disabled behind the invisible box) until it was ended in
    /// Task Manager.
    pub fn ask_graphics_restart(&mut self, choice: Option<crate::mugetsu::Graphics>) {
        // A later pick replaces an earlier one still being asked about.
        self.prompts.retain(|prompt| !prompt.is_graphics());
        self.prompts.push(Prompt::graphics(choice));
    }

    /// AQW build: offers the restart that gives frame interpolation its full speed (see
    /// `Prompt::interpolation_restart`), or takes the offer back (`wanted` false:
    /// interpolation was turned off again).
    pub fn offer_interpolation_restart(&mut self, wanted: bool) {
        self.prompts.retain(|prompt| !prompt.is_interpolation_restart());
        if wanted {
            self.prompts.push(Prompt::interpolation_restart());
        }
    }

    /// AQW build: tells the player something in the game window (see `mugetsu::Notice`).
    pub fn show_notice(&mut self, notice: crate::mugetsu::Notice) {
        self.prompts.push(Prompt::notice(notice));
    }

    pub fn show_context_menu(
        &mut self,
        menu: Vec<ruffle_core::ContextMenuItem>,
        settings: Option<crate::mugetsu::MenuSettings>,
        close_event: PlayerEvent,
    ) {
        if !menu.is_empty() || settings.is_some() {
            self.context_menu = Some(ContextMenu::new(menu, settings, close_event));
        }
    }

    pub fn close_context_menu(&mut self, player: &mut Player) {
        if let Some(context_menu) = mem::take(&mut self.context_menu) {
            player.handle_event(context_menu.close_event());
        }
    }

    pub fn is_context_menu_visible(&self) -> bool {
        self.context_menu.is_some()
    }

    /// Notifies the GUI that the player has been destroyed.
    fn on_player_destroyed(&mut self) {
        self.dialogs.close_dialogs_with_notifiers();
    }

    /// Notifies the GUI that a new player was created.
    fn on_player_created(
        &mut self,
        opt: LaunchOptions,
        content_descriptor: ContentDescriptor,
        mut player: MutexGuard<Player>,
    ) {
        self.menu_bar.currently_opened = Some((content_descriptor.clone(), opt.clone()));

        // Update dialog state to reflect the newly-opened movie's options.
        self.dialogs
            .recreate_open_dialog(opt, Some(content_descriptor), self.event_loop.clone());

        player.set_volume(self.dialogs.volume_controls.get_volume());
    }
}

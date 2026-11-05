package flash.ui {
    [Ruffle(Abstract)]
    public final class Multitouch {
        // FIXME - implement a setter and gett
        public static var inputMode:String = MultitouchInputMode.GESTURE;

        // Turu: true on a touch screen (the frontend's `UiBackend::supports_touch_events`).
        public static native function get supportsTouchEvents():Boolean;

        // Turu: Android reports ten.
        public static function get maxTouchPoints():int {
            return supportsTouchEvents ? 10 : 0;
        }
    }
}

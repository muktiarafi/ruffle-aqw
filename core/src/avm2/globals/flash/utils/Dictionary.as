package flash.utils {
    import __ruffle__.stub_constructor;

    [Ruffle(InstanceAllocator)]
    public dynamic class Dictionary {
        prototype.toJSON = function(r:String):* {
            return "Dictionary";
        };
        prototype.setPropertyIsEnumerable("toJSON", false);

        private native function enableWeakKeys():void;

        public function Dictionary(weakKeys:Boolean = false) {
            if (weakKeys) {
                enableWeakKeys();
            }
        }
    }
}

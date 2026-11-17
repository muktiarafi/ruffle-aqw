pub use crate::avm2::object::dictionary_allocator;

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::function::FunctionArgs;
use crate::avm2::value::Value;

/// Implements `new Dictionary(true)` (AQW build): see `DictionaryObject::finalize_weak_keys`.
pub fn enable_weak_keys<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    if let Some(dictionary) = this.as_object().and_then(|o| o.as_dictionary_object()) {
        dictionary.enable_weak_keys();
        activation.context.avm2.register_weak_dictionary(dictionary);
    }
    Ok(Value::Undefined)
}

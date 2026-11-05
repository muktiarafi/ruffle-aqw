//! `flash.ui.Multitouch` builtin (Turu)

use crate::avm2::activation::Activation;
use crate::avm2::function::FunctionArgs;
use crate::avm2::value::Value;
use crate::avm2::Error;

pub fn get_supports_touch_events<'gc>(
    activation: &mut Activation<'_, 'gc>,
    _this: Value<'gc>,
    _args: FunctionArgs<'_, 'gc>,
) -> Result<Value<'gc>, Error<'gc>> {
    Ok(activation.context.ui.supports_touch_events().into())
}

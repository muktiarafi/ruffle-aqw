//! AQW build heap inspector: readable descriptions of `Gc` values on a retention path.

use crate::avm2::object::script_object::ScriptObjectData;

/// AS3 object data types that are `#[repr(C)]` with `base: ScriptObjectData` as first field
/// (checked against every `*ObjectData` in this module), so their pointer can be read as one.
const SCRIPT_OBJECT_DATA_TYPES: &[&str] = &[
    "ScriptObjectData",
    "ArrayObjectData",
    "BitmapDataObjectData",
    "ByteArrayObjectData",
    "ClassObjectData",
    "ContentElementObjectData",
    "DateObjectData",
    "DictionaryObjectData",
    "DispatchObjectData",
    "DomainObjectData",
    "ElementFormatObjectData",
    "ErrorObjectData",
    "EventObjectData",
    "FileReferenceObjectData",
    "FontDescriptionObjectData",
    "FontObjectData",
    "FunctionObjectData",
    "LoaderInfoObjectData",
    "LocalConnectionObjectData",
    "MessageChannelObjectData",
    "NamespaceObjectData",
    "NetConnectionObjectData",
    "NetStreamObjectData",
    "ProxyObjectData",
    "QNameObjectData",
    "RegExpObjectData",
    "ResponderObjectData",
    "SecurityDomainObjectData",
    "ShaderDataObjectData",
    "SharedObjectObjectData",
    "SocketObjectData",
    "SoundObjectData",
    "SoundChannelObjectData",
    "SoundTransformObjectData",
    "StageObjectData",
    "StyleSheetObjectData",
    "TabStopObjectData",
    "TextBlockObjectData",
    "TextFormatObjectData",
    "TextureObjectData",
    "VectorObjectData",
    "WorkerDomainObjectData",
    "WorkerObjectData",
    "XmlListObjectData",
    "XmlObjectData",
];

/// Describes the value at `ptr`, and which of its fields holds `next` when known.
///
/// # Safety
///
/// `ptr` must be a live `Gc` value pointer in the current arena (strongly reachable).
pub(crate) unsafe fn describe<'gc>(ptr: *const (), next: Option<*const ()>) -> String {
    let full = unsafe { gc_arena::inspect::type_name(ptr) };
    let short = full.rsplit("::").next().unwrap_or(full);
    let short = short.split('<').next().unwrap_or(short);
    if full.starts_with("ruffle_core::avm2::object::") && SCRIPT_OBJECT_DATA_TYPES.contains(&short)
    {
        let data = unsafe { &*(ptr as *const ScriptObjectData<'gc>) };
        format!("{short} {}", data.debug_describe(next))
    } else {
        full.to_string()
    }
}

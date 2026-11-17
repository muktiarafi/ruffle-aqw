//! Object representation for `flash.utils.Dictionary`

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::dynamic_map::DynamicKey;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject};
use crate::avm2::value::Value;
use crate::string::AvmString;
use core::fmt;
use gc_arena::collect::Trace;
use gc_arena::{Collect, Finalization, Gc, GcWeak, Mutation};
use ruffle_common::utils::HasPrefixField;
use std::cell::Cell;

/// A class instance allocator that allocates Dictionary objects.
pub fn dictionary_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);

    Ok(DictionaryObject(Gc::new(
        activation.gc(),
        DictionaryObjectData {
            base,
            weak_keys: Cell::new(false),
        },
    ))
    .into())
}

/// An object that allows associations between objects and values.
///
/// This is implemented by way of "object space", parallel to the property
/// space that ordinary properties live in. This space has no namespaces, and
/// keys are objects instead of strings.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct DictionaryObject<'gc>(pub Gc<'gc, DictionaryObjectData<'gc>>);

#[derive(Clone, Collect, Copy, Debug)]
#[collect(no_drop)]
pub struct DictionaryObjectWeak<'gc>(pub GcWeak<'gc, DictionaryObjectData<'gc>>);

impl fmt::Debug for DictionaryObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, HasPrefixField)]
#[repr(C, align(8))]
pub struct DictionaryObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc>,

    /// `new Dictionary(true)` (AQW build): object-keyed entries don't keep their key alive.
    weak_keys: Cell<bool>,
}

// SAFETY: everything this holds is traced, except object-keyed entries of weak-keyed
// dictionaries; those are revived or removed by `DictionaryObject::finalize_weak_keys` once
// marking is complete and before anything is swept (see `Player::collect_garbage`).
unsafe impl<'gc> Collect<'gc> for DictionaryObjectData<'gc> {
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {
        if self.weak_keys.get() {
            self.base.trace_without_object_keys(cc);
        } else {
            cc.trace(&self.base);
        }
    }
}

/// Finds whether any `Gc` traced through it is dead (unmarked) after marking.
struct DeadCheck<'a, 'gc> {
    fc: &'a Finalization<'gc>,
    dead: bool,
}

impl<'gc> Trace<'gc> for DeadCheck<'_, 'gc> {
    fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
        self.dead |= Gc::is_dead(self.fc, gc);
    }

    fn trace_gc_weak(&mut self, _gc: GcWeak<'gc, ()>) {}
}

/// Marks every unmarked `Gc` traced through it as reachable again.
struct Revive<'a, 'gc> {
    fc: &'a Finalization<'gc>,
    revived: bool,
}

impl<'gc> Trace<'gc> for Revive<'_, 'gc> {
    fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
        if Gc::is_dead(self.fc, gc) {
            Gc::resurrect(self.fc, gc);
            self.revived = true;
        }
    }

    fn trace_gc_weak(&mut self, _gc: GcWeak<'gc, ()>) {}
}

impl<'gc> DictionaryObject<'gc> {
    /// Turns on weak keys (`new Dictionary(true)`).
    pub fn enable_weak_keys(self) {
        self.0.weak_keys.set(true);
    }

    pub fn downgrade(self) -> DictionaryObjectWeak<'gc> {
        DictionaryObjectWeak(Gc::downgrade(self.0))
    }

    /// Ephemeron step for a weak-keyed dictionary, run once marking is complete. Revives the
    /// values of entries whose object key is reachable (returns whether anything was revived;
    /// marking must then continue and this run again, since revived values can make more keys
    /// reachable). With `remove_dead`, once nothing more gets revived, removes entries whose key
    /// is unreachable so their key and value can be collected.
    pub fn finalize_weak_keys(self, fc: &Finalization<'gc>, remove_dead: bool) -> bool {
        let mut revived = false;
        let mut dead_keys = Vec::new();
        for (key, prop) in self.base().values().iter() {
            let DynamicKey::Object(key_object) = key else {
                continue;
            };
            let mut check = DeadCheck { fc, dead: false };
            key_object.trace(&mut check);
            if check.dead {
                dead_keys.push(*key_object);
            } else {
                let mut revive = Revive { fc, revived: false };
                prop.value.trace(&mut revive);
                revived |= revive.revived;
            }
        }
        if remove_dead && !revived && !dead_keys.is_empty() {
            let base = self.base();
            let mut values = base.values_mut(fc);
            for key in dead_keys {
                values.remove(&DynamicKey::Object(key));
            }
        }
        revived
    }

    /// Retrieve a value in the dictionary's object space.
    pub fn get_property_by_object(self, name: Object<'gc>) -> Value<'gc> {
        self.base()
            .values()
            .get(&DynamicKey::Object(name))
            .map(|v| v.value)
            .unwrap_or(Value::Undefined)
    }

    /// Set a value in the dictionary's object space.
    pub fn set_property_by_object(self, name: Object<'gc>, value: Value<'gc>, mc: &Mutation<'gc>) {
        self.base()
            .values_mut(mc)
            .insert(DynamicKey::Object(name), value);
    }

    /// Delete a value from the dictionary's object space.
    pub fn delete_property_by_object(self, name: Object<'gc>, mc: &Mutation<'gc>) {
        self.base().values_mut(mc).remove(&DynamicKey::Object(name));
    }

    pub fn has_property_by_object(self, name: Object<'gc>) -> bool {
        self.base().values().contains_key(&DynamicKey::Object(name))
    }
}

impl<'gc> TObject<'gc> for DictionaryObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    // Calling `setPropertyIsEnumerable` on a `Dictionary` has no effect -
    // stringified properties are always enumerable.
    fn set_local_property_is_enumerable(
        &self,
        _mc: &Mutation<'gc>,
        _name: AvmString<'gc>,
        _is_enumerable: bool,
    ) {
    }

    fn get_enumerant_value(
        self,
        index: u32,
        _activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        Ok(*self
            .base()
            .values()
            .value_at(index as usize)
            .unwrap_or(&Value::Undefined))
    }
}

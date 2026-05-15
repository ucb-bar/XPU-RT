/*
 * CompGen HAL — Shared scaffolding
 *
 * Implements the tiny amount of target-independent glue that every HAL
 * driver shares.  In particular this file provides the vtable accessor
 * declared in <compgen/hal.h>.
 */

#include "hal_internal.h"

const compgen_device_vtable_t*
compgen_device_get_vtable(compgen_device_t device)
{
    /*
     * By convention every concrete `compgen_device_s` starts with a
     * `const compgen_device_vtable_t*` member.  We simply dereference
     * the pointer.
     */
    return device->vtable;
}

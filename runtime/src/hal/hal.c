/*
 * XPU-RT HAL — Shared scaffolding
 *
 * Implements the tiny amount of target-independent glue that every HAL
 * driver shares.  In particular this file provides the vtable accessor
 * declared in <xpu_rt/hal.h>.
 */

#include "hal_internal.h"

const xpu_rt_device_vtable_t*
xpu_rt_device_get_vtable(xpu_rt_device_t device)
{
    /*
     * By convention every concrete `xpu_rt_device_s` starts with a
     * `const xpu_rt_device_vtable_t*` member.  We simply dereference
     * the pointer.
     */
    return device->vtable;
}

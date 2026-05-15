/*
 * XPU-RT HAL — Internal base struct
 *
 * Defines the minimum layout that every concrete xpu_rt_device_s must
 * satisfy: a vtable pointer as the first member.  Driver-specific headers
 * re-declare the full struct with additional fields but must keep the
 * vtable pointer first.
 *
 * This header is private to the HAL implementation.
 */

#ifndef XPU_RT_HAL_INTERNAL_H
#define XPU_RT_HAL_INTERNAL_H

#include "xpu_rt/hal.h"

struct xpu_rt_device_s {
    const xpu_rt_device_vtable_t* vtable;
};

#endif /* XPU_RT_HAL_INTERNAL_H */

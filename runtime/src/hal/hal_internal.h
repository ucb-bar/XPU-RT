/*
 * CompGen HAL — Internal base struct
 *
 * Defines the minimum layout that every concrete compgen_device_s must
 * satisfy: a vtable pointer as the first member.  Driver-specific headers
 * re-declare the full struct with additional fields but must keep the
 * vtable pointer first.
 *
 * This header is private to the HAL implementation.
 */

#ifndef COMPGEN_HAL_INTERNAL_H
#define COMPGEN_HAL_INTERNAL_H

#include "compgen/hal.h"

struct compgen_device_s {
    const compgen_device_vtable_t* vtable;
};

#endif /* COMPGEN_HAL_INTERNAL_H */

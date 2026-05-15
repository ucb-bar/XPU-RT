/*
 * CompGen HAL Types
 *
 * Opaque handle types, status codes, and info-key enumerations used across
 * the entire HAL surface.  This header is intentionally free of function
 * declarations so that lightweight consumers (e.g. generated code that only
 * needs the types) do not pull in the full vtable definition.
 */

#ifndef COMPGEN_HAL_TYPES_H
#define COMPGEN_HAL_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum compgen_status_t {
    COMPGEN_STATUS_OK = 0,
    COMPGEN_STATUS_ERROR = 1,
    COMPGEN_STATUS_OUT_OF_MEMORY = 2,
    COMPGEN_STATUS_UNIMPLEMENTED = 3,
    COMPGEN_STATUS_INVALID_ARGUMENT = 4,
    COMPGEN_STATUS_NOT_FOUND = 5,
} compgen_status_t;

/* ------------------------------------------------------------------ */
/* Device info query keys                                              */
/* ------------------------------------------------------------------ */

typedef enum compgen_device_info_key_t {
    COMPGEN_DEVICE_INFO_DEVICE_TYPE = 0,
    COMPGEN_DEVICE_INFO_MEMORY_TOTAL = 1,
    COMPGEN_DEVICE_INFO_COMPUTE_UNITS = 2,
    COMPGEN_DEVICE_INFO_MAX_DISPATCH_CONCURRENCY = 3,
    COMPGEN_DEVICE_INFO_ADDRESS_SPACE_COUNT = 4,
    COMPGEN_DEVICE_INFO_SUPPORTS_ASYNC_DMA = 5,
    COMPGEN_DEVICE_INFO_MAX_ALLOC_SIZE = 6,
} compgen_device_info_key_t;

/* ------------------------------------------------------------------ */
/* Opaque handle types                                                 */
/* ------------------------------------------------------------------ */

typedef struct compgen_device_s*     compgen_device_t;
typedef struct compgen_buffer_s*     compgen_buffer_t;
typedef struct compgen_executable_s* compgen_executable_t;

#ifdef __cplusplus
}
#endif

#endif /* COMPGEN_HAL_TYPES_H */

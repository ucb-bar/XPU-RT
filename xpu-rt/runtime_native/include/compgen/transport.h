/*
 * CompGen Transport — C-level inter-node communication vtable.
 *
 * Provides a pluggable transport API for data and command exchange
 * between runtime nodes.  Backends register their vtable and the
 * executor dispatches through it.
 *
 * Backends:
 *   - Local:       direct function call + buffer copy (same process).
 *   - Zephyr IPC:  k_msgq / k_pipe / ipc_service.
 *   - Network:     stub for gRPC/TCP (protocol defined, impl deferred).
 */

#ifndef COMPGEN_TRANSPORT_H_
#define COMPGEN_TRANSPORT_H_

#include <stddef.h>
#include <stdint.h>

#include "compgen/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Message structure                                                   */
/* ------------------------------------------------------------------ */

typedef struct cg_transport_msg {
    int32_t  tag;            /* application-defined tag for multiplexing */
    void    *payload;        /* raw payload buffer */
    size_t   payload_size;   /* payload size in bytes */
} cg_transport_msg_t;

/* ------------------------------------------------------------------ */
/* Transport handle (opaque)                                           */
/* ------------------------------------------------------------------ */

typedef struct cg_transport cg_transport_t;

/* ------------------------------------------------------------------ */
/* Transport vtable                                                    */
/* ------------------------------------------------------------------ */

typedef struct cg_transport_vtable {
    /** Open the transport channel. */
    cg_status_t (*open)(cg_transport_t *self);

    /** Close the transport channel and release resources. */
    void (*close)(cg_transport_t *self);

    /**
     * Send a message.
     *
     * @param timeout_us  Timeout in microseconds.  0 = non-blocking,
     *                    UINT64_MAX = block indefinitely.
     * @return CG_STATUS_OK on success.
     */
    cg_status_t (*send)(cg_transport_t *self,
                         const cg_transport_msg_t *msg,
                         uint64_t timeout_us);

    /**
     * Receive a message.
     *
     * @param out_msg     On success, filled with the received message.
     *                    Caller must free payload after use.
     * @param timeout_us  Timeout in microseconds.  0 = non-blocking.
     * @return CG_STATUS_OK on success, CG_STATUS_UNAVAILABLE on timeout.
     */
    cg_status_t (*recv)(cg_transport_t *self,
                         cg_transport_msg_t *out_msg,
                         uint64_t timeout_us);

    /**
     * Barrier — block until all participants reach this point.
     */
    cg_status_t (*barrier)(cg_transport_t *self);

    /** Return the transport name (static string). */
    const char *(*name)(const cg_transport_t *self);
} cg_transport_vtable_t;

/* ------------------------------------------------------------------ */
/* Transport base structure (embed in backend structs)                 */
/* ------------------------------------------------------------------ */

struct cg_transport {
    const cg_transport_vtable_t *vtable;
    int                          is_open;
};

/* ------------------------------------------------------------------ */
/* Convenience inline dispatchers                                      */
/* ------------------------------------------------------------------ */

static inline cg_status_t cg_transport_open(cg_transport_t *t) {
    return t->vtable->open(t);
}

static inline void cg_transport_close(cg_transport_t *t) {
    t->vtable->close(t);
}

static inline cg_status_t cg_transport_send(cg_transport_t *t,
                                              const cg_transport_msg_t *msg,
                                              uint64_t timeout_us) {
    return t->vtable->send(t, msg, timeout_us);
}

static inline cg_status_t cg_transport_recv(cg_transport_t *t,
                                              cg_transport_msg_t *out_msg,
                                              uint64_t timeout_us) {
    return t->vtable->recv(t, out_msg, timeout_us);
}

static inline cg_status_t cg_transport_barrier(cg_transport_t *t) {
    return t->vtable->barrier(t);
}

static inline const char *cg_transport_name(const cg_transport_t *t) {
    return t->vtable->name(t);
}

/* ------------------------------------------------------------------ */
/* Local transport constructor                                         */
/* ------------------------------------------------------------------ */

/**
 * Create a local (in-process) transport.
 *
 * @param queue_depth  Maximum number of messages in the queue.
 * @param out          On success, receives the transport pointer.
 * @return CG_STATUS_OK on success.
 */
cg_status_t cg_transport_local_create(size_t queue_depth,
                                       cg_transport_t **out);

/**
 * Destroy a local transport and free resources.
 */
void cg_transport_local_destroy(cg_transport_t *t);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* COMPGEN_TRANSPORT_H_ */

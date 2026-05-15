/*
 * CompGen Local Transport — in-process message queue.
 *
 * A simple FIFO queue for same-process communication between
 * runtime components.  Thread-safe via mutex.
 */

#include "compgen/transport.h"

#include <stdlib.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__)
  #include <pthread.h>
  #define USE_PTHREADS 1
#else
  #define USE_PTHREADS 0
#endif

/* ------------------------------------------------------------------ */
/* Local transport structure                                           */
/* ------------------------------------------------------------------ */

typedef struct {
    cg_transport_t       base;
    cg_transport_msg_t  *queue;
    size_t               capacity;
    size_t               head;
    size_t               tail;
    size_t               count;
#if USE_PTHREADS
    pthread_mutex_t      mutex;
    pthread_cond_t       cond;
#endif
} cg_local_transport_t;

/* ------------------------------------------------------------------ */
/* Vtable implementations                                              */
/* ------------------------------------------------------------------ */

static cg_status_t _local_open(cg_transport_t *self) {
    self->is_open = 1;
    return CG_STATUS_OK;
}

static void _local_close(cg_transport_t *self) {
    cg_local_transport_t *lt = (cg_local_transport_t *)self;
    self->is_open = 0;

    /* Free any remaining message payloads */
#if USE_PTHREADS
    pthread_mutex_lock(&lt->mutex);
#endif

    while (lt->count > 0) {
        cg_transport_msg_t *msg = &lt->queue[lt->head];
        if (msg->payload) {
            free(msg->payload);
            msg->payload = NULL;
        }
        lt->head = (lt->head + 1) % lt->capacity;
        lt->count--;
    }

#if USE_PTHREADS
    pthread_cond_broadcast(&lt->cond);
    pthread_mutex_unlock(&lt->mutex);
#endif
}

static cg_status_t _local_send(cg_transport_t *self,
                                 const cg_transport_msg_t *msg,
                                 uint64_t timeout_us) {
    cg_local_transport_t *lt = (cg_local_transport_t *)self;
    (void)timeout_us;

    if (!self->is_open) return CG_STATUS_FAILED_PRECONDITION;

#if USE_PTHREADS
    pthread_mutex_lock(&lt->mutex);
#endif

    if (lt->count >= lt->capacity) {
#if USE_PTHREADS
        pthread_mutex_unlock(&lt->mutex);
#endif
        return CG_STATUS_OUT_OF_RANGE;
    }

    /* Copy the message (deep-copy payload) */
    cg_transport_msg_t *slot = &lt->queue[lt->tail];
    slot->tag = msg->tag;
    slot->payload_size = msg->payload_size;
    if (msg->payload && msg->payload_size > 0) {
        slot->payload = malloc(msg->payload_size);
        if (!slot->payload) {
#if USE_PTHREADS
            pthread_mutex_unlock(&lt->mutex);
#endif
            return CG_STATUS_RESOURCE_EXHAUSTED;
        }
        memcpy(slot->payload, msg->payload, msg->payload_size);
    } else {
        slot->payload = NULL;
    }

    lt->tail = (lt->tail + 1) % lt->capacity;
    lt->count++;

#if USE_PTHREADS
    pthread_cond_signal(&lt->cond);
    pthread_mutex_unlock(&lt->mutex);
#endif

    return CG_STATUS_OK;
}

static cg_status_t _local_recv(cg_transport_t *self,
                                 cg_transport_msg_t *out_msg,
                                 uint64_t timeout_us) {
    cg_local_transport_t *lt = (cg_local_transport_t *)self;

    if (!self->is_open) return CG_STATUS_FAILED_PRECONDITION;

#if USE_PTHREADS
    pthread_mutex_lock(&lt->mutex);

    /* Wait for a message */
    if (lt->count == 0 && timeout_us > 0) {
        if (timeout_us == UINT64_MAX) {
            /* Block indefinitely */
            while (lt->count == 0 && self->is_open) {
                pthread_cond_wait(&lt->cond, &lt->mutex);
            }
        } else {
            /* Timed wait */
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += (long)(timeout_us / 1000000ULL);
            ts.tv_nsec += (long)((timeout_us % 1000000ULL) * 1000);
            if (ts.tv_nsec >= 1000000000L) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000L;
            }
            while (lt->count == 0 && self->is_open) {
                int rc = pthread_cond_timedwait(&lt->cond, &lt->mutex, &ts);
                if (rc != 0) break;  /* timeout or error */
            }
        }
    }

    if (lt->count == 0) {
        pthread_mutex_unlock(&lt->mutex);
        return CG_STATUS_UNAVAILABLE;
    }
#else
    if (lt->count == 0) {
        return CG_STATUS_UNAVAILABLE;
    }
#endif

    /* Dequeue */
    *out_msg = lt->queue[lt->head];
    lt->queue[lt->head].payload = NULL;  /* ownership transferred */
    lt->head = (lt->head + 1) % lt->capacity;
    lt->count--;

#if USE_PTHREADS
    pthread_mutex_unlock(&lt->mutex);
#endif

    return CG_STATUS_OK;
}

static cg_status_t _local_barrier(cg_transport_t *self) {
    (void)self;
    return CG_STATUS_OK;  /* single-participant: no-op */
}

static const char *_local_name(const cg_transport_t *self) {
    (void)self;
    return "local";
}

static const cg_transport_vtable_t _local_vtable = {
    .open    = _local_open,
    .close   = _local_close,
    .send    = _local_send,
    .recv    = _local_recv,
    .barrier = _local_barrier,
    .name    = _local_name,
};

/* ------------------------------------------------------------------ */
/* Constructor / destructor                                            */
/* ------------------------------------------------------------------ */

cg_status_t cg_transport_local_create(size_t queue_depth,
                                       cg_transport_t **out) {
    if (!out) return CG_STATUS_INVALID_ARGUMENT;
    if (queue_depth == 0) queue_depth = 64;

    cg_local_transport_t *lt = (cg_local_transport_t *)calloc(
        1, sizeof(cg_local_transport_t));
    if (!lt) return CG_STATUS_RESOURCE_EXHAUSTED;

    lt->queue = (cg_transport_msg_t *)calloc(
        queue_depth, sizeof(cg_transport_msg_t));
    if (!lt->queue) {
        free(lt);
        return CG_STATUS_RESOURCE_EXHAUSTED;
    }

    lt->base.vtable  = &_local_vtable;
    lt->base.is_open = 0;
    lt->capacity     = queue_depth;
    lt->head         = 0;
    lt->tail         = 0;
    lt->count        = 0;

#if USE_PTHREADS
    pthread_mutex_init(&lt->mutex, NULL);
    pthread_cond_init(&lt->cond, NULL);
#endif

    *out = &lt->base;
    return CG_STATUS_OK;
}

void cg_transport_local_destroy(cg_transport_t *t) {
    if (!t) return;
    cg_local_transport_t *lt = (cg_local_transport_t *)t;

    if (t->is_open) {
        _local_close(t);
    }

    free(lt->queue);

#if USE_PTHREADS
    pthread_mutex_destroy(&lt->mutex);
    pthread_cond_destroy(&lt->cond);
#endif

    free(lt);
}

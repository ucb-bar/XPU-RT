/*
 * CompGen Topology — C-level runtime topology descriptor.
 *
 * Mirrors the Python RuntimeTopology in a C-friendly layout.
 * Used by the generated runtime code to discover nodes, devices,
 * and inter-node links at startup.
 *
 * The topology is built at codegen time and baked into the runtime
 * binary as a const structure.
 */

#ifndef COMPGEN_TOPOLOGY_H_
#define COMPGEN_TOPOLOGY_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Enums                                                               */
/* ------------------------------------------------------------------ */

typedef enum cg_deployment_kind {
    CG_DEPLOY_SINGLE_DEVICE   = 0,
    CG_DEPLOY_MULTI_DEVICE    = 1,
    CG_DEPLOY_MULTI_DOMAIN    = 2,  /* heterogeneous SoC */
    CG_DEPLOY_DISTRIBUTED     = 3,
} cg_deployment_kind_t;

typedef enum cg_node_role {
    CG_NODE_ROLE_HOST         = 0,
    CG_NODE_ROLE_ACCELERATOR  = 1,
    CG_NODE_ROLE_WORKER       = 2,
    CG_NODE_ROLE_COORDINATOR  = 3,
} cg_node_role_t;

typedef enum cg_runtime_env {
    CG_RUNTIME_LINUX          = 0,
    CG_RUNTIME_ZEPHYR         = 1,
    CG_RUNTIME_BARE_METAL     = 2,
    CG_RUNTIME_FIRMWARE       = 3,
} cg_runtime_env_t;

typedef enum cg_transport_kind {
    CG_TRANSPORT_LOCAL        = 0,
    CG_TRANSPORT_SHARED_MEM   = 1,
    CG_TRANSPORT_ZEPHYR_IPC   = 2,
    CG_TRANSPORT_DMA          = 3,
    CG_TRANSPORT_PCIE         = 4,
    CG_TRANSPORT_NETWORK      = 5,
    CG_TRANSPORT_CUSTOM       = 6,
} cg_transport_kind_t;

/* ------------------------------------------------------------------ */
/* Device descriptor                                                   */
/* ------------------------------------------------------------------ */

typedef struct cg_device_desc {
    int         device_index;   /* index into global device array */
    const char *device_type;    /* "cpu", "gpu", "npu", "dsp", ... */
    const char *name;           /* human-readable name */
} cg_device_desc_t;

/* ------------------------------------------------------------------ */
/* Node descriptor                                                     */
/* ------------------------------------------------------------------ */

typedef struct cg_node_desc {
    const char          *name;
    cg_node_role_t       role;
    cg_runtime_env_t     runtime_env;
    const cg_device_desc_t *devices;
    size_t                  num_devices;
} cg_node_desc_t;

/* ------------------------------------------------------------------ */
/* Link descriptor                                                     */
/* ------------------------------------------------------------------ */

typedef struct cg_link_desc {
    const char          *src_node;
    const char          *dst_node;
    cg_transport_kind_t  transport;
    float                bandwidth_gbps;
    float                latency_us;
    int                  bidirectional;
} cg_link_desc_t;

/* ------------------------------------------------------------------ */
/* Topology descriptor                                                 */
/* ------------------------------------------------------------------ */

typedef struct cg_topology {
    cg_deployment_kind_t   deployment;
    const cg_node_desc_t  *nodes;
    size_t                 num_nodes;
    const cg_link_desc_t  *links;
    size_t                 num_links;
} cg_topology_t;

/* ------------------------------------------------------------------ */
/* Query helpers                                                       */
/* ------------------------------------------------------------------ */

/**
 * Find a node by name.
 *
 * @return Pointer to the node descriptor, or NULL.
 */
const cg_node_desc_t *cg_topology_find_node(const cg_topology_t *topo,
                                              const char *name);

/**
 * Find the node that owns a given device index.
 *
 * @return Pointer to the node descriptor, or NULL.
 */
const cg_node_desc_t *cg_topology_node_for_device(const cg_topology_t *topo,
                                                     int device_index);

/**
 * Find the link between two nodes.
 *
 * @return Pointer to the link descriptor, or NULL.
 */
const cg_link_desc_t *cg_topology_find_link(const cg_topology_t *topo,
                                              const char *src_node,
                                              const char *dst_node);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* COMPGEN_TOPOLOGY_H_ */

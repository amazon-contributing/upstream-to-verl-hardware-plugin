/* Copyright (c) 2026 BAAI. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 * Count native device-to-host transfers, including eager CPU fallbacks.
 */
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <nrt/nrt_async.h>

static _Atomic unsigned long reads, bytes;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static __typeof__(&nrt_tensor_read) real_read;
static __typeof__(&nrta_tensor_read) real_async_read;

static void load_runtime(void) {
    void *runtime = dlopen("libnrt.so.1", RTLD_LAZY);
    if (!runtime) abort();
    real_read = dlsym(runtime, "nrt_tensor_read");
    real_async_read = dlsym(runtime, "nrta_tensor_read");
    if (!real_read || !real_async_read) abort();
}

unsigned long weight_sync_reads(void) { return reads; }
unsigned long weight_sync_bytes(void) { return bytes; }

static void record(size_t size) {
    atomic_fetch_add(&reads, 1);
    atomic_fetch_add(&bytes, size);
}

NRT_STATUS nrt_tensor_read(const nrt_tensor_t *tensor, void *buf, size_t offset, size_t size) {
    pthread_once(&once, load_runtime);
    record(size);
    return real_read(tensor, buf, offset, size);
}

NRT_STATUS nrta_tensor_read(void *buf, nrt_tensor_t *tensor, uint64_t offset, uint64_t size,
                            int lnc, int queue, NRT_STATUS *ret, nrta_seq_t *seq) {
    pthread_once(&once, load_runtime);
    record(size);
    return real_async_read(buf, tensor, offset, size, lnc, queue, ret, seq);
}

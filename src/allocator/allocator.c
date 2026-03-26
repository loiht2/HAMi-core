#include "allocator.h"
#include "include/log_utils.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>

/* --- Memory Resizer UDS Client --- */

#define RESIZE_SOCKET_PATH "/var/run/hami/memory-resizer.sock"
#define RESIZE_TIMEOUT_SEC 5
#define RESIZE_INCREMENT   (512ULL * 1024 * 1024)  /* 512 MiB */
#define RESIZE_PROTOCOL_VERSION 1

/* Wire format: must match Go server exactly */
typedef struct {
    uint32_t version;              /* 0:4   */
    int32_t  device_id;            /* 4:8   */
    uint64_t current_usage;        /* 8:16  */
    uint64_t current_limit;        /* 16:24 */
    uint64_t requested_increase;   /* 24:32 */
    char     cache_file[256];      /* 32:288 */
} resize_request_t;

typedef struct {
    uint32_t status;     /* 0=resized, 1=denied, 2=error */
    uint64_t new_limit;
} resize_response_t;

/*
 * request_memory_resize: Contact the memory-resizer DaemonSet via UDS
 * to request additional GPU memory for this container.
 * Returns 0 on success (limit was increased), -1 on failure.
 */
static int request_memory_resize(int dev, uint64_t current_usage, uint64_t current_limit) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        LOG_ERROR("resize: socket() failed errno=%d", errno);
        return -1;
    }

    /* Set send/recv timeout */
    struct timeval tv;
    tv.tv_sec = RESIZE_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, RESIZE_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("resize: connect(%s) failed errno=%d", RESIZE_SOCKET_PATH, errno);
        close(sock);
        return -1;
    }

    /* Build request */
    resize_request_t req;
    memset(&req, 0, sizeof(req));
    req.version = RESIZE_PROTOCOL_VERSION;
    req.device_id = dev;
    req.current_usage = current_usage;
    req.current_limit = current_limit;
    req.requested_increase = RESIZE_INCREMENT;

    char* cache_env = getenv("CUDA_DEVICE_MEMORY_SHARED_CACHE");
    if (cache_env != NULL) {
        strncpy(req.cache_file, cache_env, sizeof(req.cache_file) - 1);
    }

    if (send(sock, &req, sizeof(req), 0) != sizeof(req)) {
        LOG_ERROR("resize: send() failed errno=%d", errno);
        close(sock);
        return -1;
    }

    /* Receive response */
    resize_response_t resp;
    memset(&resp, 0, sizeof(resp));
    ssize_t n = recv(sock, &resp, sizeof(resp), 0);
    close(sock);

    if (n != sizeof(resp)) {
        LOG_ERROR("resize: recv() got %zd bytes, expected %zu, errno=%d", n, sizeof(resp), errno);
        return -1;
    }

    if (resp.status == 0) {
        LOG_INFO("resize: success, new_limit=%lu", resp.new_limit);
        return 0;
    }

    LOG_WARN("resize: denied status=%u", resp.status);
    return -1;
}

/* --- End Memory Resizer Client --- */

size_t BITSIZE = 512;
size_t IPCSIZE = 2097152;
size_t OVERSIZE = 134217728;
//int pidfound;

region_list *r_list;
allocated_list *device_overallocated;
allocated_list *device_allocasync;

#define ALIGN       2097152
#define MULTI_PARAM 1

#define CHUNK_SIZE  (OVERSIZE/BITSIZE)
#define __CHUNK_SIZE__  CHUNK_SIZE

extern size_t initial_offset;
extern CUresult
    cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data);
extern CUresult cuMemoryFree(CUdeviceptr dptr);

pthread_once_t allocator_allocate_flag = PTHREAD_ONCE_INIT;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

size_t round_up(size_t size, size_t unit) {
    if (size & (unit-1))
        return ((size / unit) + 1 ) * unit;
    return size;
}

int oom_check(const int dev, size_t addon) {
    int count1=0;
    CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetCount,&count1);
    CUdevice d;
    if (dev==-1)
        cuCtxGetDevice(&d);
    else
        d=dev;
    uint64_t limit = get_current_device_memory_limit(d);
    // Use real NVML-reported memory usage instead of internally tracked value
    // This ensures OOM is triggered based on actual GPU memory consumption
    size_t _usage = get_gpu_memory_real_usage(d);

    if (limit == 0) {
        return 0;
    }

    size_t new_allocated = _usage + addon;
    LOG_INFO("_usage=%lu limit=%lu new_allocated=%lu",_usage,limit,new_allocated);
    if (new_allocated > limit) {
        LOG_WARN("Device %d approaching OOM %lu / %lu, requesting resize",
                 d, new_allocated, limit);

        /* Keep requesting resizes until the limit is high enough or denied */
        int max_attempts = 20;  /* safety cap: 20 * 512MiB = 10GiB max growth */
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            uint64_t cur_limit = get_current_device_memory_limit(d);
            if (new_allocated <= cur_limit) {
                LOG_INFO("Device %d resize OK after %d attempt(s): limit=%lu usage=%lu",
                         d, attempt, cur_limit, new_allocated);
                return 0;  /* OOM avoided */
            }
            if (request_memory_resize(d, _usage, cur_limit) != 0) {
                LOG_WARN("Device %d resize denied at attempt %d", d, attempt + 1);
                break;
            }
        }

        uint64_t final_limit = get_current_device_memory_limit(d);
        if (new_allocated <= final_limit) {
            return 0;  /* OOM avoided after loop */
        }

        LOG_ERROR("Device %d OOM %lu / %lu", d, new_allocated, final_limit);

        if (clear_proc_slot_nolock(1) > 0)
            return oom_check(dev,addon);
        return 1;
    }
    return 0;
}

CUresult view_vgpu_allocator() {
    allocated_list_entry *al;
    size_t total;
    total=0;
    LOG_INFO("[view1]:overallocated:");
    for (al=device_overallocated->head;al!=NULL;al=al->next){
        LOG_INFO("(%p %lu)\t",(void *)al->entry->address,al->entry->length);
        total+=al->entry->length;
    }
    LOG_INFO("total=%lu",total);
    size_t t = get_current_device_memory_usage(0);
    LOG_INFO("current_device_memory_usage:%lu",t);
    return 0;
}

CUresult get_listsize(allocated_list *al, size_t *size) {
    if (al->length == 0){
        *size = 0;
        return CUDA_SUCCESS;
    }
    size_t count=0;
    allocated_list_entry *val;
    for (val=al->head;val!=NULL;val=val->next){
        count+=val->entry->length;
    }
    *size = count;
    return CUDA_SUCCESS;
}

void allocator_init() {
    LOG_DEBUG("Allocator_init\n");
    
    device_overallocated = malloc(sizeof(allocated_list));
    LIST_INIT(device_overallocated);
    device_allocasync=malloc(sizeof(allocated_list));
    LIST_INIT(device_allocasync);

    pthread_mutex_init(&mutex,NULL);
}

int add_chunk(CUdeviceptr *address, size_t size) {
    size_t addr=0;
    size_t allocsize;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size))
        return CUDA_ERROR_OUT_OF_MEMORY;
    
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    if (size <= IPCSIZE)
        res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAlloc_v2,&e->entry->address,size);
    else{
        e->entry->length = size;
        res = cuMemoryAllocate(&e->entry->address, size, e->entry->allocHandle);
    }
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        return res;
    }
    LIST_ADD(device_overallocated,e);
    //uint64_t t_size;
    *address = e->entry->address;
    allocsize = size;
    cuCtxGetDevice(&dev);
    add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    return 0;
}

int add_chunk_only(CUdeviceptr address, size_t size) {
    pthread_mutex_lock(&mutex);
    size_t addr=0;
    size_t allocsize;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size)){
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    LIST_ADD(device_overallocated,e);
    //uint64_t t_size;
    e->entry->address=address;
    allocsize = size;
    cuCtxGetDevice(&dev);
    add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    pthread_mutex_unlock(&mutex);
    return 0;
}

int check_memory_type(CUdeviceptr address) {
    allocated_list_entry *cursor;
    cursor = device_overallocated->head;
    for (cursor=device_overallocated->head;cursor!=NULL;cursor=cursor->next){
        if ((cursor->entry->address <= address) && (cursor->entry->address+cursor->entry->length>=address))
            return CU_MEMORYTYPE_DEVICE;
    }
    return CU_MEMORYTYPE_HOST;
}

int remove_chunk(allocated_list *a_list, CUdeviceptr dptr) {
    size_t t_size;
    if (a_list->length==0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val=a_list->head;val!=NULL;val=val->next){
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            cuMemoryFree(dptr);
            LIST_REMOVE(a_list,val);
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
            return 0;
        }
    }
    return -1;
}

int remove_chunk_only(CUdeviceptr dptr) {
    allocated_list *a_list = device_overallocated;
    size_t t_size;
    if (a_list->length == 0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size = val->entry->length;
            LIST_REMOVE(a_list, val);
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
            return 0;
        }
    }
    return -1;
}

int allocate_raw(CUdeviceptr *dptr, size_t size) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk(dptr, size);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int free_raw(CUdeviceptr dptr) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk(device_overallocated, dptr);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int remove_chunk_async(
    allocated_list *a_list, CUdeviceptr dptr, CUstream hStream) {
    size_t t_size;
    if (a_list->length == 0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeAsync,dptr,hStream);
            LIST_REMOVE(a_list,val);
            a_list->limit-=t_size;
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(),dev,t_size,2);
            return 0;
        }
    }
    return -1;
}

int free_raw_async(CUdeviceptr dptr, CUstream hStream) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk_async(device_allocasync, dptr, hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int add_chunk_async(CUdeviceptr *address, size_t size, CUstream hStream) {
    size_t addr=0;
    size_t allocsize;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size))
        return -1;

    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocAsync,&e->entry->address,size,hStream);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        return res;
    }
    *address = e->entry->address;
    CUmemoryPool pool;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetMemPool,&pool,dev);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuDeviceGetMemPool failed res=%d",res);
        return res;
    }
    size_t poollimit;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAttribute,pool,CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,&poollimit);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemPoolGetAttribute failed res=%d",res);
        return res;
    }
    if (poollimit != 0) {
        if (poollimit> device_allocasync->limit) {
            allocsize = (poollimit-device_allocasync->limit < size)? poollimit-device_allocasync->limit : size;
            cuCtxGetDevice(&dev);
            add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
            device_allocasync->limit=device_allocasync->limit+allocsize;
            e->entry->length=allocsize;
        }else{
            e->entry->length=0;
        } 
    }
    LIST_ADD(device_allocasync,e);
    return 0;
}

int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk_async(dptr,size,hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

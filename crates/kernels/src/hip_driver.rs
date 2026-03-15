use crate::types::{GpuDriverError, GpuDriverResult};
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::{Arc, Mutex};

pub type HipError = c_int;
pub type HipDevice = c_int;
pub type HipContext = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipDevicePtr = *mut c_void;
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;

#[derive(Debug, Clone)]
pub struct HipDeviceProperties {
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_grid_size: [i32; 3],
    pub max_block_size: [i32; 3],
    pub clock_rate: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub max_threads_per_multiprocessor: i32,
    pub multiprocessor_count: i32,
}

pub struct HipDriverFunctions {
    pub hip_init: Symbol<'static, unsafe extern "C" fn(flags: c_uint) -> HipError>,
    pub hip_device_get_count: Symbol<'static, unsafe extern "C" fn(*mut c_int) -> HipError>,
    pub hip_device_get: Symbol<'static, unsafe extern "C" fn(*mut HipDevice, c_int) -> HipError>,
    pub hip_device_get_name: Symbol<'static, unsafe extern "C" fn(*mut c_char, c_int, HipDevice) -> HipError>,
    pub hip_device_get_attribute: Symbol<'static, unsafe extern "C" fn(*mut c_int, c_int, HipDevice) -> HipError>,
    pub hip_ctx_create: Symbol<'static, unsafe extern "C" fn(*mut HipContext, c_uint, HipDevice) -> HipError>,
    pub hip_ctx_destroy: Symbol<'static, unsafe extern "C" fn(HipContext) -> HipError>,
    pub hip_ctx_set_current: Symbol<'static, unsafe extern "C" fn(HipContext) -> HipError>,
    pub hip_ctx_get_current: Symbol<'static, unsafe extern "C" fn(*mut HipContext) -> HipError>,
    pub hip_malloc: Symbol<'static, unsafe extern "C" fn(*mut HipDevicePtr, usize) -> HipError>,
    pub hip_free: Symbol<'static, unsafe extern "C" fn(HipDevicePtr) -> HipError>,
    pub hip_memcpy_h2d: Symbol<'static, unsafe extern "C" fn(HipDevicePtr, *const c_void, usize) -> HipError>,
    pub hip_memcpy_d2h: Symbol<'static, unsafe extern "C" fn(*mut c_void, HipDevicePtr, usize) -> HipError>,
    pub hip_memcpy_h2d_async: Symbol<'static, unsafe extern "C" fn(HipDevicePtr, *const c_void, usize, HipStream) -> HipError>,
    pub hip_memcpy_d2h_async: Symbol<'static, unsafe extern "C" fn(*mut c_void, HipDevicePtr, usize, HipStream) -> HipError>,
    pub hip_memset: Symbol<'static, unsafe extern "C" fn(HipDevicePtr, c_int, usize) -> HipError>,
    pub hip_module_load_data: Symbol<'static, unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError>,
    pub hip_module_unload: Symbol<'static, unsafe extern "C" fn(HipModule) -> HipError>,
    pub hip_module_get_function: Symbol<'static, unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError>,
    pub hip_launch_kernel: Symbol<'static, unsafe extern "C" fn(HipFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, HipStream, *mut *mut c_void, *mut *mut c_void) -> HipError>,
    pub hip_stream_create: Symbol<'static, unsafe extern "C" fn(*mut HipStream) -> HipError>,
    pub hip_stream_destroy: Symbol<'static, unsafe extern "C" fn(HipStream) -> HipError>,
    pub hip_stream_synchronize: Symbol<'static, unsafe extern "C" fn(HipStream) -> HipError>,
    pub hip_device_synchronize: Symbol<'static, unsafe extern "C" fn() -> HipError>,
    pub hip_event_create: Symbol<'static, unsafe extern "C" fn(*mut HipEvent) -> HipError>,
    pub hip_event_destroy: Symbol<'static, unsafe extern "C" fn(HipEvent) -> HipError>,
    pub hip_event_record: Symbol<'static, unsafe extern "C" fn(HipEvent, HipStream) -> HipError>,
    pub hip_event_synchronize: Symbol<'static, unsafe extern "C" fn(HipEvent) -> HipError>,
    pub hip_event_elapsed_time: Symbol<'static, unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> HipError>,
    pub hip_get_error_string: Symbol<'static, unsafe extern "C" fn(HipError) -> *const c_char>,
}

pub struct HipDriverInterface {
    driver_lib: Library,
    functions: HipDriverFunctions,
    context: HipContext,
    device_properties: HipDeviceProperties,
    module_cache: Arc<Mutex<HashMap<String, HipModule>>>,
}

impl HipDriverInterface {
    pub fn new(device_id: i32) -> GpuDriverResult<Self> {
        let driver_lib = unsafe {
            #[cfg(target_os = "linux")]
            let lib_names = vec!["libamdhip64.so", "libamdhip64.so.4", "libamdhip64.so.5"];
            #[cfg(target_os = "windows")]
            let lib_names = vec!["amdhip64.dll"];
            #[cfg(target_os = "macos")]
            let lib_names = vec!["libamdhip64.dylib"];

            let mut lib = None;
            for name in lib_names {
                if let Ok(loaded_lib) = Library::new(name) {
                    lib = Some(loaded_lib);
                    break;
                }
            }

            lib.ok_or_else(|| GpuDriverError::DriverLoadError("Failed to load HIP driver library".into()))?
        };

        let functions = unsafe {
            HipDriverFunctions {
                hip_init: driver_lib.get(b"hipInit\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipInit: {}", e)))?,
                hip_device_get_count: driver_lib.get(b"hipGetDeviceCount\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipGetDeviceCount: {}", e)))?,
                hip_device_get: driver_lib.get(b"hipGetDevice\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipGetDevice: {}", e)))?,
                hip_device_get_name: driver_lib.get(b"hipGetDeviceProperties\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipGetDeviceProperties: {}", e)))?,
                hip_device_get_attribute: driver_lib.get(b"hipDeviceGetAttribute\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipDeviceGetAttribute: {}", e)))?,
                hip_ctx_create: driver_lib.get(b"hipCtxCreate\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipCtxCreate: {}", e)))?,
                hip_ctx_destroy: driver_lib.get(b"hipCtxDestroy\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipCtxDestroy: {}", e)))?,
                hip_ctx_set_current: driver_lib.get(b"hipCtxSetCurrent\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipCtxSetCurrent: {}", e)))?,
                hip_ctx_get_current: driver_lib.get(b"hipCtxGetCurrent\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipCtxGetCurrent: {}", e)))?,
                hip_malloc: driver_lib.get(b"hipMalloc\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipMalloc: {}", e)))?,
                hip_free: driver_lib.get(b"hipFree\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipFree: {}", e)))?,
                hip_memcpy_h2d: driver_lib.get(b"hipMemcpy\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipMemcpy: {}", e)))?,
                hip_memcpy_d2h: driver_lib.get(b"hipMemcpy\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipMemcpy: {}", e)))?,
                hip_memcpy_h2d_async: driver_lib.get(b"hipMemcpyAsync\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipMemcpyAsync: {}", e)))?,
                hip_memcpy_d2h_async: driver_lib.get(b"hipMemcpyAsync\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipMemcpyAsync: {}", e)))?,
                hip_memset: driver_lib.get(b"hipMemset\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipMemset: {}", e)))?,
                hip_module_load_data: driver_lib.get(b"hipModuleLoadData\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipModuleLoadData: {}", e)))?,
                hip_module_unload: driver_lib.get(b"hipModuleUnload\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipModuleUnload: {}", e)))?,
                hip_module_get_function: driver_lib.get(b"hipModuleGetFunction\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipModuleGetFunction: {}", e)))?,
                hip_launch_kernel: driver_lib.get(b"hipModuleLaunchKernel\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipModuleLaunchKernel: {}", e)))?,
                hip_stream_create: driver_lib.get(b"hipStreamCreate\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipStreamCreate: {}", e)))?,
                hip_stream_destroy: driver_lib.get(b"hipStreamDestroy\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipStreamDestroy: {}", e)))?,
                hip_stream_synchronize: driver_lib.get(b"hipStreamSynchronize\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipStreamSynchronize: {}", e)))?,
                hip_device_synchronize: driver_lib.get(b"hipDeviceSynchronize\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipDeviceSynchronize: {}", e)))?,
                hip_event_create: driver_lib.get(b"hipEventCreate\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipEventCreate: {}", e)))?,
                hip_event_destroy: driver_lib.get(b"hipEventDestroy\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipEventDestroy: {}", e)))?,
                hip_event_record: driver_lib.get(b"hipEventRecord\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipEventRecord: {}", e)))?,
                hip_event_synchronize: driver_lib.get(b"hipEventSynchronize\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipEventSynchronize: {}", e)))?,
                hip_event_elapsed_time: driver_lib.get(b"hipEventElapsedTime\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipEventElapsedTime: {}", e)))?,
                hip_get_error_string: driver_lib.get(b"hipGetErrorString\0")
                    .map_err(|e| GpuDriverError::FunctionLoadError(format!("hipGetErrorString: {}", e)))?,
            }
        };

        unsafe {
            let result = (functions.hip_init)(0);
            if result != 0 {
                return Err(GpuDriverError::InitializationError(format!("HIP initialization failed with error: {}", result)));
            }
        }

        let mut device = 0;
        unsafe {
            let result = (functions.hip_device_get)(&mut device, device_id);
            if result != 0 {
                return Err(GpuDriverError::DeviceError(format!("Failed to get HIP device {}: {}", device_id, result)));
            }
        }

        let mut context = ptr::null_mut();
        unsafe {
            let result = (functions.hip_ctx_create)(&mut context, 0, device);
            if result != 0 {
                return Err(GpuDriverError::ContextError(format!("Failed to create HIP context: {}", result)));
            }
        }

        let device_properties = Self::get_device_properties(&functions, device)?;

        Ok(HipDriverInterface {
            driver_lib,
            functions,
            context,
            device_properties,
            module_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn get_device_properties(functions: &HipDriverFunctions, device: HipDevice) -> GpuDriverResult<HipDeviceProperties> {
        let mut name_buf = [0u8; 256];
        let mut value = 0;

        unsafe {
            let result = (functions.hip_device_get_name)(name_buf.as_mut_ptr() as *mut c_char, 256, device);
            if result != 0 {
                return Err(GpuDriverError::DeviceError(format!("Failed to get device name: {}", result)));
            }
        }

        let name = unsafe {
            CStr::from_ptr(name_buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned()
        };

        let get_attribute = |attr: c_int| -> GpuDriverResult<i32> {
            let mut val = 0;
            unsafe {
                let result = (functions.hip_device_get_attribute)(&mut val, attr, device);
                if result != 0 {
                    return Err(GpuDriverError::DeviceError(format!("Failed to get device attribute {}: {}", attr, result)));
                }
            }
            Ok(val)
        };

        Ok(HipDeviceProperties {
            name,
            compute_capability_major: get_attribute(75)?,  // hipDeviceAttributeComputeCapabilityMajor
            compute_capability_minor: get_attribute(76)?,  // hipDeviceAttributeComputeCapabilityMinor
            total_global_mem: get_attribute(1)? as usize,  // hipDeviceAttributeTotalGlobalMem
            shared_mem_per_block: get_attribute(8)? as usize, // hipDeviceAttributeMaxSharedMemoryPerBlock
            regs_per_block: get_attribute(12)?,             // hipDeviceAttributeMaxRegistersPerBlock
            warp_size: get_attribute(10)?,                  // hipDeviceAttributeWarpSize
            max_threads_per_block: get_attribute(11)?,     // hipDeviceAttributeMaxThreadsPerBlock
            max_grid_size: [
                get_attribute(5)?,                          // hipDeviceAttributeMaxGridDimX
                get_attribute(6)?,                          // hipDeviceAttributeMaxGridDimY
                get_attribute(7)?,                          // hipDeviceAttributeMaxGridDimZ
            ],
            max_block_size: [
                get_attribute(2)?,                          // hipDeviceAttributeMaxBlockDimX
                get_attribute(3)?,                          // hipDeviceAttributeMaxBlockDimY
                get_attribute(4)?,                          // hipDeviceAttributeMaxBlockDimZ
            ],
            clock_rate: get_attribute(9)?,                  // hipDeviceAttributeClockRate
            memory_clock_rate: get_attribute(36)?,          // hipDeviceAttributeMemoryClockRate
            memory_bus_width: get_attribute(37)?,           // hipDeviceAttributeMemoryBusWidth
            l2_cache_size: get_attribute(38)?,              // hipDeviceAttributeL2CacheSize
            max_threads_per_multiprocessor: get_attribute(39)?, // hipDeviceAttributeMaxThreadsPerMultiProcessor
            multiprocessor_count: get_attribute(16)?,       // hipDeviceAttributeMultiprocessorCount
        })
    }

    pub fn allocate_memory(&self, size: usize) -> GpuDriverResult<HipDevicePtr> {
        let mut ptr = ptr::null_mut();
        unsafe {
            let result = (self.functions.hip_malloc)(&mut ptr, size);
            if result != 0 {
                return Err(GpuDriverError::AllocationError(format!("HIP memory allocation failed: {}", result)));
            }
        }
        Ok(ptr)
    }

    pub fn free_memory(&self, ptr: HipDevicePtr) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_free)(ptr);
            if result != 0 {
                return Err(GpuDriverError::AllocationError(format!("HIP memory free failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn copy_host_to_device(&self, dst: HipDevicePtr, src: *const c_void, size: usize) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_memcpy_h2d)(dst, src, size);
            if result != 0 {
                return Err(GpuDriverError::MemoryError(format!("Host to device copy failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn copy_device_to_host(&self, dst: *mut c_void, src: HipDevicePtr, size: usize) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_memcpy_d2h)(dst, src, size);
            if result != 0 {
                return Err(GpuDriverError::MemoryError(format!("Device to host copy failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn copy_host_to_device_async(&self, dst: HipDevicePtr, src: *const c_void, size: usize, stream: HipStream) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_memcpy_h2d_async)(dst, src, size, stream);
            if result != 0 {
                return Err(GpuDriverError::MemoryError(format!("Async host to device copy failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn memset(&self, ptr: HipDevicePtr, value: i32, size: usize) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_memset)(ptr, value, size);
            if result != 0 {
                return Err(GpuDriverError::MemoryError(format!("Memory set failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn load_module(&self, module_data: &[u8]) -> GpuDriverResult<HipModule> {
        let mut module = ptr::null_mut();
        unsafe {
            let result = (self.functions.hip_module_load_data)(&mut module, module_data.as_ptr() as *const c_void);
            if result != 0 {
                return Err(GpuDriverError::ModuleError(format!("Module load failed: {}", result)));
            }
        }
        Ok(module)
    }

    pub fn get_function(&self, module: HipModule, function_name: &str) -> GpuDriverResult<HipFunction> {
        let function_name_c = CString::new(function_name)
            .map_err(|e| GpuDriverError::InvalidParameter(format!("Invalid function name: {}", e)))?;

        let mut function = ptr::null_mut();
        unsafe {
            let result = (self.functions.hip_module_get_function)(&mut function, module, function_name_c.as_ptr());
            if result != 0 {
                return Err(GpuDriverError::ModuleError(format!("Function '{}' not found: {}", function_name, result)));
            }
        }
        Ok(function)
    }

    pub fn launch_kernel(
        &self,
        function: HipFunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: HipStream,
        kernel_params: &mut [*mut c_void],
    ) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_launch_kernel)(
                function,
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                shared_mem_bytes,
                stream,
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            );
            if result != 0 {
                return Err(GpuDriverError::ExecutionError(format!("Kernel launch failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn create_stream(&self) -> GpuDriverResult<HipStream> {
        let mut stream = ptr::null_mut();
        unsafe {
            let result = (self.functions.hip_stream_create)(&mut stream);
            if result != 0 {
                return Err(GpuDriverError::StreamError(format!("Stream creation failed: {}", result)));
            }
        }
        Ok(stream)
    }

    pub fn destroy_stream(&self, stream: HipStream) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_stream_destroy)(stream);
            if result != 0 {
                return Err(GpuDriverError::StreamError(format!("Stream destruction failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn synchronize_stream(&self, stream: HipStream) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_stream_synchronize)(stream);
            if result != 0 {
                return Err(GpuDriverError::SynchronizationError(format!("Stream synchronization failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn synchronize_device(&self) -> GpuDriverResult<()> {
        unsafe {
            let result = (self.functions.hip_device_synchronize)();
            if result != 0 {
                return Err(GpuDriverError::SynchronizationError(format!("Device synchronization failed: {}", result)));
            }
        }
        Ok(())
    }

    pub fn get_device_properties(&self) -> &HipDeviceProperties {
        &self.device_properties
    }

    pub fn cache_module(&self, key: String, module: HipModule) {
        let mut cache = self.module_cache.lock().unwrap();
        cache.insert(key, module);
    }

    pub fn get_cached_module(&self, key: &str) -> Option<HipModule> {
        let cache = self.module_cache.lock().unwrap();
        cache.get(key).copied()
    }

    fn get_error_string(&self, error: HipError) -> String {
        unsafe {
            let error_ptr = (self.functions.hip_get_error_string)(error);
            if error_ptr.is_null() {
                format!("Unknown HIP error: {}", error)
            } else {
                CStr::from_ptr(error_ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl Drop for HipDriverInterface {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.functions.hip_ctx_destroy)(self.context);
        }
    }
}

unsafe impl Send for HipDriverInterface {}
unsafe impl Sync for HipDriverInterface {}
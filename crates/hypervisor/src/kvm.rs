//! KVM implementation

use std::os::unix::io::RawFd;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::mem;

// KVM constants
const KVM_GET_API_VERSION: u64 = 0xAE00;
const KVM_CREATE_VM: u64 = 0xAE01;
const KVM_CREATE_VCPU: u64 = 0xAE41;
const KVM_SET_TSS_ADDR: u64 = 0xAE47;
const KVM_SET_IDENTITY_MAP_ADDR: u64 = 0xAE48;
const KVM_SET_USER_MEMORY_REGION: u64 = 0x4020AE46;
const KVM_RUN: u64 = 0xAE80;
const KVM_GET_REGS: u64 = 0x8090AE82;
const KVM_SET_REGS: u64 = 0x4090AE83;
const KVM_GET_SREGS: u64 = 0x8138AE84;
const KVM_SET_SREGS: u64 = 0x4138AE85;
const KVM_GET_VCPU_MMAP_SIZE: u64 = 0xAE04;

// KVM exit reasons
const KVM_EXIT_HLT: u32 = 1;
const KVM_EXIT_IO: u32 = 2;
const KVM_EXIT_MMIO: u32 = 3;
const KVM_EXIT_INTR: u32 = 4;
const KVM_EXIT_SHUTDOWN: u32 = 5;
const KVM_EXIT_FAIL_ENTRY: u32 = 6;
const KVM_EXIT_INTERNAL_ERROR: u32 = 7;

#[repr(C)]
struct KvmUserMemoryRegion {
    slot: u32,
    flags: u32,
    guest_phys_addr: u64,
    memory_size: u64,
    userspace_addr: u64,
}

#[repr(C)]
struct KvmRun {
    request_interrupt_window: u8,
    immediate_exit: u8,
    padding1: [u8; 6],
    exit_reason: u32,
    ready_for_interrupt_injection: u8,
    if_flag: u8,
    flags: u16,
    cr8: u64,
    apic_base: u64,
    padding2: [u8; 8],
    kvm_valid_regs: u64,
    kvm_dirty_regs: u64,
    s: KvmRunUnion,
}

#[repr(C)]
union KvmRunUnion {
    hlt: KvmHlt,
    io: KvmIo,
    mmio: KvmMmio,
    interrupt: KvmInterrupt,
    fail_entry: KvmFailEntry,
    internal_error: KvmInternalError,
    padding: [u8; 256],
}

#[repr(C)]
struct KvmHlt {
    _unused: u8,
}

#[repr(C)]
struct KvmIo {
    direction: u8,
    size: u8,
    port: u16,
    count: u32,
    data_offset: u64,
}

#[repr(C)]
struct KvmMmio {
    phys_addr: u64,
    data: [u8; 8],
    len: u32,
    is_write: u8,
}

#[repr(C)]
struct KvmInterrupt {
    irq: i32,
}

#[repr(C)]
struct KvmFailEntry {
    hardware_entry_failure_reason: u64,
}

#[repr(C)]
struct KvmInternalError {
    suberror: u32,
    ndata: u32,
    data: [u64; 16],
}

#[repr(C)]
struct KvmRegs {
    rax: u64,
    rbx: u64,
    rcx: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    rsp: u64,
    rbp: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rip: u64,
    rflags: u64,
}

/// KVM implementation
pub struct Kvm {
    kvm_fd: RawFd,
    vm_fd: RawFd,
}

impl Kvm {
    /// Create a new KVM instance
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Open /dev/kvm
        let kvm_file = File::open("/dev/kvm")?;
        let kvm_fd = kvm_file.as_raw_fd();
        
        // Check KVM API version
        let api_version = unsafe {
            libc::ioctl(kvm_fd, KVM_GET_API_VERSION, 0)
        };
        
        if api_version < 0 {
            return Err("Failed to get KVM API version".into());
        }
        
        // Create a VM
        let vm_fd = unsafe {
            libc::ioctl(kvm_fd, KVM_CREATE_VM, 0)
        };
        
        if vm_fd < 0 {
            return Err("Failed to create VM".into());
        }
        
        Ok(Self {
            kvm_fd,
            vm_fd,
        })
    }
    
    /// Initialize the VM
    pub fn init_vm(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Set up identity map address (required for x86_64)
        let identity_map_addr = 0xfffff000u64;
        let result = unsafe {
            libc::ioctl(self.vm_fd, KVM_SET_IDENTITY_MAP_ADDR, &identity_map_addr as *const u64)
        };
        
        if result < 0 {
            return Err("Failed to set identity map address".into());
        }
        
        // Set up TSS address (required for x86_64)
        let tss_addr = 0xfffbd000u64;
        let result = unsafe {
            libc::ioctl(self.vm_fd, KVM_SET_TSS_ADDR, &tss_addr as *const u64)
        };
        
        if result < 0 {
            return Err("Failed to set TSS address".into());
        }
        
        Ok(())
    }
    
    /// Set up memory region for the VM
    pub fn setup_memory_region(&self, slot: u32, guest_phys_addr: u64, 
                                memory_size: u64, userspace_addr: u64) -> Result<(), Box<dyn std::error::Error>> {
        let region = KvmUserMemoryRegion {
            slot,
            flags: 0, // KVM_MEM_LOG_DIRTY_PAGES if needed
            guest_phys_addr,
            memory_size,
            userspace_addr,
        };
        
        let result = unsafe {
            libc::ioctl(self.vm_fd, KVM_SET_USER_MEMORY_REGION, &region as *const KvmUserMemoryRegion)
        };
        
        if result < 0 {
            return Err("Failed to set user memory region".into());
        }
        
        Ok(())
    }
    
    /// Create a VCPU
    pub fn create_vcpu(&self, id: u8) -> Result<Vcpu, Box<dyn std::error::Error>> {
        Vcpu::new(self.vm_fd, id)
    }
    
    /// Get the KVM file descriptor
    pub fn kvm_fd(&self) -> RawFd {
        self.kvm_fd
    }
    
    /// Get the VM file descriptor
    pub fn vm_fd(&self) -> RawFd {
        self.vm_fd
    }
}

/// VCPU representation
pub struct Vcpu {
    id: u8,
    vcpu_fd: RawFd,
    run_data: *mut KvmRun,
    run_data_size: usize,
}

impl Vcpu {
    /// Create a new VCPU
    pub fn new(vm_fd: RawFd, id: u8) -> Result<Self, Box<dyn std::error::Error>> {
        let vcpu_fd = unsafe {
            libc::ioctl(vm_fd, KVM_CREATE_VCPU, id as u64)
        };
        
        if vcpu_fd < 0 {
            return Err("Failed to create VCPU".into());
        }
        
        // Get the size of the KVM run structure
        let run_data_size = unsafe {
            libc::ioctl(vcpu_fd, KVM_GET_VCPU_MMAP_SIZE, 0)
        };
        
        if run_data_size < 0 {
            return Err("Failed to get VCPU mmap size".into());
        }
        
        // Map the run data structure
        let run_data = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                run_data_size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                vcpu_fd,
                0,
            )
        };
        
        if run_data == libc::MAP_FAILED {
            return Err("Failed to mmap VCPU run data".into());
        }
        
        Ok(Self {
            id,
            vcpu_fd,
            run_data: run_data as *mut KvmRun,
            run_data_size: run_data_size as usize,
        })
    }
    
    /// Initialize VCPU registers
    pub fn init_regs(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut regs = KvmRegs {
            rax: 0,
            rbx: 0,
            rcx: 0,
            rdx: 0,
            rsi: 0,
            rdi: 0,
            rsp: 0x1000, // Stack pointer
            rbp: 0,
            r8: 0,
            r9: 0,
            r10: 0,
            r11: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
            rip: 0x1000, // Instruction pointer
            rflags: 0x2, // Interrupt flag
        };
        
        let result = unsafe {
            libc::ioctl(self.vcpu_fd, KVM_SET_REGS, &regs as *const KvmRegs)
        };
        
        if result < 0 {
            return Err("Failed to set VCPU registers".into());
        }
        
        Ok(())
    }
    
    /// Run the VCPU
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let result = unsafe {
            libc::ioctl(self.vcpu_fd, KVM_RUN, 0)
        };
        
        if result < 0 {
            return Err("Failed to run VCPU".into());
        }
        
        // Handle the exit reason
        let run_data = unsafe { &*self.run_data };
        match run_data.exit_reason {
            KVM_EXIT_HLT => {
                // Guest executed HLT instruction
                Ok(())
            },
            KVM_EXIT_IO => {
                // Handle I/O operation
                self.handle_io_exit()
            },
            KVM_EXIT_MMIO => {
                // Handle MMIO operation
                self.handle_mmio_exit()
            },
            KVM_EXIT_INTR => {
                // Interrupt occurred
                Ok(())
            },
            KVM_EXIT_SHUTDOWN => {
                // Guest shutdown
                Ok(())
            },
            KVM_EXIT_FAIL_ENTRY => {
                Err("VCPU entry failed".into())
            },
            KVM_EXIT_INTERNAL_ERROR => {
                Err("Internal KVM error".into())
            },
            _ => {
                Err("Unknown exit reason".into())
            }
        }
    }
    
    /// Handle I/O exit
    fn handle_io_exit(&self) -> Result<(), Box<dyn std::error::Error>> {
        // For now, just acknowledge the I/O
        // In a real implementation, this would handle actual I/O operations
        Ok(())
    }
    
    /// Handle MMIO exit
    fn handle_mmio_exit(&self) -> Result<(), Box<dyn std::error::Error>> {
        // For now, just acknowledge the MMIO
        // In a real implementation, this would handle memory-mapped I/O
        Ok(())
    }
    
    /// Get the VCPU file descriptor
    pub fn vcpu_fd(&self) -> RawFd {
        self.vcpu_fd
    }
    
    /// Get the run data pointer
    pub fn run_data(&self) -> *mut KvmRun {
        self.run_data
    }
}

impl Drop for Vcpu {
    fn drop(&mut self) {
        if !self.run_data.is_null() {
            unsafe {
                libc::munmap(self.run_data as *mut libc::c_void, self.run_data_size);
            }
        }
    }
}
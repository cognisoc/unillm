//! Hypervisor crate for KVM-based virtualization

mod kvm;
mod vfio;
mod memory;

pub use kvm::{Kvm, Vcpu};
pub use vfio::Vfio;
pub use memory::{MemoryManager, PinnedMemory};

/// Hypervisor implementation
pub struct Hypervisor {
    kvm: Kvm,
    memory_manager: MemoryManager,
}

impl Hypervisor {
    /// Create a new hypervisor instance
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let kvm = Kvm::new()?;
        let memory_manager = MemoryManager::new();
        Ok(Self { kvm, memory_manager })
    }
    
    /// Initialize the hypervisor
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.kvm.init_vm()
    }
    
    /// Create a VCPU
    pub fn create_vcpu(&self, id: u8) -> Result<Vcpu, Box<dyn std::error::Error>> {
        self.kvm.create_vcpu(id)
    }
    
    /// Set up VFIO passthrough for a device
    pub fn setup_vfio_passthrough(&self, device_path: &str) -> Result<Vfio, Box<dyn std::error::Error>> {
        let vfio = Vfio::new(device_path)?;
        vfio.enable()?;
        Ok(vfio)
    }
    
    /// Allocate pinned memory
    pub fn alloc_pinned_memory(&self, size: usize) -> Result<PinnedMemory, Box<dyn std::error::Error>> {
        self.memory_manager.alloc_pinned(size)
    }
}
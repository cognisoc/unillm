//! HAL (Hardware Abstraction Layer) crate for PCIe/IOMMU/MSI-X handling

mod pci;
mod iommu;
mod msix;

pub use pci::PciDevice;
pub use iommu::Iommu;
pub use msix::Msix;

/// Hardware abstraction layer implementation
pub struct HardwareAbstractionLayer {
    pci_device: PciDevice,
    iommu: Iommu,
    msix: Msix,
}

impl HardwareAbstractionLayer {
    /// Create a new HAL instance
    pub fn new(bus: u8, device: u8, function: u8) -> Self {
        Self {
            pci_device: PciDevice::new(bus, device, function),
            iommu: Iommu::new(),
            msix: Msix::new(),
        }
    }
    
    /// Initialize the hardware abstraction layer
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.pci_device.init()?;
        self.iommu.init()?;
        self.msix.init()?;
        Ok(())
    }
}
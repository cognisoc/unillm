//! VFIO implementation for device passthrough

use std::os::unix::io::RawFd;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

// VFIO constants
const VFIO_GROUP_PATH: &str = "/dev/vfio/vfio";
const VFIO_CONTAINER_PATH: &str = "/dev/vfio/vfio";

// VFIO ioctl commands
const VFIO_GET_API_VERSION: u64 = 0x3E01;
const VFIO_CHECK_EXTENSION: u64 = 0x3E02;
const VFIO_SET_IOMMU: u64 = 0x3E03;
const VFIO_GROUP_SET_CONTAINER: u64 = 0x3E04;
const VFIO_GROUP_UNSET_CONTAINER: u64 = 0x3E05;
const VFIO_GROUP_GET_STATUS: u64 = 0x3E06;
const VFIO_GROUP_SET_STATUS: u64 = 0x3E07;
const VFIO_GROUP_GET_DEVICE_FD: u64 = 0x3E08;
const VFIO_DEVICE_GET_INFO: u64 = 0x3E09;
const VFIO_DEVICE_GET_REGION_INFO: u64 = 0x3E0A;
const VFIO_DEVICE_GET_IRQ_INFO: u64 = 0x3E0B;
const VFIO_DEVICE_SET_IRQS: u64 = 0x3E0C;
const VFIO_DEVICE_RESET: u64 = 0x3E0D;
const VFIO_IOMMU_MAP_DMA: u64 = 0x3E0E;
const VFIO_IOMMU_UNMAP_DMA: u64 = 0x3E0F;

// VFIO flags
const VFIO_GROUP_FLAGS_VIABLE: u32 = 1 << 0;
const VFIO_GROUP_FLAGS_CONTAINER_SET: u32 = 1 << 1;

// IOMMU types
const VFIO_TYPE1_IOMMU: u32 = 1;
const VFIO_TYPE1v2_IOMMU: u32 = 2;

#[repr(C)]
struct VfioGroupStatus {
    argsz: u32,
    flags: u32,
}

#[repr(C)]
struct VfioDeviceInfo {
    argsz: u32,
    flags: u32,
    num_regions: u32,
    num_irqs: u32,
}

#[repr(C)]
struct VfioRegionInfo {
    argsz: u32,
    flags: u32,
    index: u32,
    cap_offset: u32,
    size: u64,
    offset: u64,
}

#[repr(C)]
struct VfioIommuType1DmaMap {
    argsz: u32,
    flags: u32,
    vaddr: u64,
    iova: u64,
    size: u64,
}

/// VFIO implementation
pub struct Vfio {
    container_fd: RawFd,
    group_fd: RawFd,
    device_fd: RawFd,
}

impl Vfio {
    /// Create a new VFIO instance for a device
    pub fn new(device_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Open the VFIO container
        let container_file = File::open(VFIO_CONTAINER_PATH)?;
        let container_fd = container_file.as_raw_fd();
        
        // Extract group number from device path
        // Device path is typically in the form /dev/vfio/<group_num>
        let path = Path::new(device_path);
        let group_name = path.file_name()
            .ok_or("Invalid device path")?
            .to_str()
            .ok_or("Invalid device path")?;
        
        let group_num: u32 = group_name.parse()
            .map_err(|_| "Invalid group number")?;
        
        // Open the VFIO group
        let group_path = format!("/dev/vfio/{}", group_num);
        let group_file = File::open(&group_path)?;
        let group_fd = group_file.as_raw_fd();
        
        // Set the container for the group
        // This would typically use ioctl(VFIO_GROUP_SET_CONTAINER)
        // For now, we'll just store the file descriptors
        
        // Open the VFIO device
        let device_file = File::open(device_path)?;
        let device_fd = device_file.as_raw_fd();
        
        Ok(Self {
            container_fd,
            group_fd,
            device_fd,
        })
    }
    
    /// Enable the device for passthrough
    pub fn enable(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Check if the group is viable
        let mut status = VfioGroupStatus {
            argsz: std::mem::size_of::<VfioGroupStatus>() as u32,
            flags: 0,
        };
        
        let result = unsafe {
            libc::ioctl(self.group_fd, VFIO_GROUP_GET_STATUS, &mut status as *mut VfioGroupStatus)
        };
        
        if result < 0 {
            return Err("Failed to get group status".into());
        }
        
        if (status.flags & VFIO_GROUP_FLAGS_VIABLE) == 0 {
            return Err("Group is not viable".into());
        }
        
        // Set the container for the group
        let result = unsafe {
            libc::ioctl(self.group_fd, VFIO_GROUP_SET_CONTAINER, &self.container_fd as *const RawFd)
        };
        
        if result < 0 {
            return Err("Failed to set container for group".into());
        }
        
        // Set the IOMMU type
        let result = unsafe {
            libc::ioctl(self.container_fd, VFIO_SET_IOMMU, &VFIO_TYPE1_IOMMU as *const u32)
        };
        
        if result < 0 {
            return Err("Failed to set IOMMU type".into());
        }
        
        // Enable the group
        let mut status = VfioGroupStatus {
            argsz: std::mem::size_of::<VfioGroupStatus>() as u32,
            flags: VFIO_GROUP_FLAGS_CONTAINER_SET,
        };
        
        let result = unsafe {
            libc::ioctl(self.group_fd, VFIO_GROUP_SET_STATUS, &status as *const VfioGroupStatus)
        };
        
        if result < 0 {
            return Err("Failed to enable group".into());
        }
        
        Ok(())
    }
    
    /// Map a memory region to the device
    pub fn map_region(&self, guest_addr: u64, size: usize) -> Result<u64, Box<dyn std::error::Error>> {
        let dma_map = VfioIommuType1DmaMap {
            argsz: std::mem::size_of::<VfioIommuType1DmaMap>() as u32,
            flags: 0,
            vaddr: guest_addr,
            iova: guest_addr, // Use same address for simplicity
            size: size as u64,
        };
        
        let result = unsafe {
            libc::ioctl(self.container_fd, VFIO_IOMMU_MAP_DMA, &dma_map as *const VfioIommuType1DmaMap)
        };
        
        if result < 0 {
            return Err("Failed to map DMA region".into());
        }
        
        Ok(guest_addr)
    }
    
    /// Unmap a memory region from the device
    pub fn unmap_region(&self, guest_addr: u64, size: usize) -> Result<(), Box<dyn std::error::Error>> {
        let dma_unmap = VfioIommuType1DmaMap {
            argsz: std::mem::size_of::<VfioIommuType1DmaMap>() as u32,
            flags: 0,
            vaddr: guest_addr,
            iova: guest_addr,
            size: size as u64,
        };
        
        let result = unsafe {
            libc::ioctl(self.container_fd, VFIO_IOMMU_UNMAP_DMA, &dma_unmap as *const VfioIommuType1DmaMap)
        };
        
        if result < 0 {
            return Err("Failed to unmap DMA region".into());
        }
        
        Ok(())
    }
    
    /// Get device information
    pub fn get_device_info(&self) -> Result<VfioDeviceInfo, Box<dyn std::error::Error>> {
        let mut info = VfioDeviceInfo {
            argsz: std::mem::size_of::<VfioDeviceInfo>() as u32,
            flags: 0,
            num_regions: 0,
            num_irqs: 0,
        };
        
        let result = unsafe {
            libc::ioctl(self.device_fd, VFIO_DEVICE_GET_INFO, &mut info as *mut VfioDeviceInfo)
        };
        
        if result < 0 {
            return Err("Failed to get device info".into());
        }
        
        Ok(info)
    }
    
    /// Get region information
    pub fn get_region_info(&self, index: u32) -> Result<VfioRegionInfo, Box<dyn std::error::Error>> {
        let mut info = VfioRegionInfo {
            argsz: std::mem::size_of::<VfioRegionInfo>() as u32,
            flags: 0,
            index,
            cap_offset: 0,
            size: 0,
            offset: 0,
        };
        
        let result = unsafe {
            libc::ioctl(self.device_fd, VFIO_DEVICE_GET_REGION_INFO, &mut info as *mut VfioRegionInfo)
        };
        
        if result < 0 {
            return Err("Failed to get region info".into());
        }
        
        Ok(info)
    }
    
    /// Get the container file descriptor
    pub fn container_fd(&self) -> RawFd {
        self.container_fd
    }
    
    /// Get the group file descriptor
    pub fn group_fd(&self) -> RawFd {
        self.group_fd
    }
    
    /// Get the device file descriptor
    pub fn device_fd(&self) -> RawFd {
        self.device_fd
    }
}
// Simple build test to check if the project compiles

fn main() {
    println!("Testing basic compilation...");
    
    // Test hypervisor
    match unillm_hypervisor::Hypervisor::new() {
        Ok(mut hv) => {
            println!("Hypervisor created successfully");
            if let Err(e) = hv.init() {
                println!("Hypervisor init failed: {}", e);
            } else {
                println!("Hypervisor initialized successfully");
            }
        },
        Err(e) => println!("Hypervisor creation failed: {}", e),
    }
    
    // Test HAL
    let mut pci_device = unillm_hal::PciDevice::new(0, 0, 0);
    match pci_device.init() {
        Ok(_) => println!("PCI device initialized successfully"),
        Err(e) => println!("PCI device init failed: {}", e),
    }
    
    println!("Basic compilation test completed");
}
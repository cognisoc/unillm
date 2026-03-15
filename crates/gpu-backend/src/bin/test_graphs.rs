//! Test program for graph capture functionality

use gpu_backend::{DecodeGraph, Operation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing graph capture functionality...");
    
    // Create a new graph
    let mut graph = DecodeGraph::new();
    
    println!("Created graph with {} nodes", graph.node_count());
    
    // Add some nodes to the graph
    let matmul_node = graph.add_node(
        Operation::MatMul,
        vec![], // No inputs for this test
        vec![], // No outputs for this test
    );
    println!("Added MatMul node with ID: {}", matmul_node);
    
    let add_node = graph.add_node(
        Operation::Add,
        vec![matmul_node], // Input from matmul node
        vec![],            // No outputs for this test
    );
    println!("Added Add node with ID: {}", add_node);
    
    println!("Graph now has {} nodes", graph.node_count());
    
    // Try to capture the graph
    match graph.capture() {
        Ok(()) => println!("Graph captured successfully!"),
        Err(e) => println!("Failed to capture graph: {}", e),
    }
    
    println!("Graph captured status: {}", graph.is_captured());
    
    Ok(())
}
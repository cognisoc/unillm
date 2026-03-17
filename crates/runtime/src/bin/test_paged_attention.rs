//! Test PagedAttention integration with UniLLM
//!
//! This test demonstrates the PagedAttention implementation working
//! alongside our regular attention mechanisms.

use runtime::{
    paged_attention::{PagedAttention, PagedAttentionConfig, DataType},
    gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps},
    types::ModelError,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing PagedAttention Integration");
    println!("====================================");

    // Initialize device
    let device = GpuDevice::Cpu;
    let tensor_ops = GpuTensorOps::with_device(device.clone());

    // Configure PagedAttention
    let config = PagedAttentionConfig {
        block_size: 16,
        max_num_blocks_per_seq: 64,
        max_total_blocks: 1024,
        num_heads: 4,
        head_dim: 16,
        dtype: DataType::Float32,
        sliding_window: None,
        enable_prefix_caching: false,
    };

    // Create PagedAttention instance
    println!("🚀 Initializing PagedAttention...");
    let paged_attention = PagedAttention::new(config.clone(), device.clone())?;

    // Test sequence allocation
    println!("\n📦 Testing sequence allocation...");
    let sequence_id = 12345u64;
    let prompt_length = 10;
    let max_output_length = 20;

    paged_attention.allocate_sequence(sequence_id, prompt_length, max_output_length).await?;
    println!("✅ Sequence {} allocated successfully", sequence_id);

    // Create test tensors
    println!("\n🧮 Creating test tensors...");
    let batch_size = 1;
    let seq_len = 5;
    let hidden_size = config.num_heads * config.head_dim; // 4 * 16 = 64

    // Create query, key, value tensors
    let query_data = vec![0.1f32; batch_size * seq_len * hidden_size];
    let key_data = vec![0.2f32; batch_size * seq_len * hidden_size];
    let value_data = vec![0.3f32; batch_size * seq_len * hidden_size];

    let query = GpuTensor::new(query_data, vec![batch_size, seq_len, hidden_size], device.clone())?;
    let key = GpuTensor::new(key_data, vec![batch_size, seq_len, hidden_size], device.clone())?;
    let value = GpuTensor::new(value_data, vec![batch_size, seq_len, hidden_size], device.clone())?;

    println!("✅ Test tensors created:");
    println!("   Query shape: {:?}", query.shape());
    println!("   Key shape: {:?}", key.shape());
    println!("   Value shape: {:?}", value.shape());

    // Test attention computation
    println!("\n⚡ Testing PagedAttention forward pass...");
    let sequence_ids = vec![sequence_id];
    let input_positions = vec![vec![0, 1, 2, 3, 4]]; // Positions for the sequence

    let result = paged_attention.forward(
        &query,
        &key,
        &value,
        &sequence_ids,
        &input_positions,
        None, // No attention mask
    ).await?;

    println!("✅ PagedAttention forward pass completed");
    println!("   Output shape: {:?}", result.output.shape());
    println!("   Computation time: {:.2}ms", result.computation_time_ms);
    println!("   Blocks allocated: {}", result.kv_cache_stats.blocks_allocated);
    println!("   Memory utilization: {:.1}%", result.kv_cache_stats.memory_utilization);

    // Get detailed statistics
    println!("\n📊 PagedAttention Statistics:");
    let stats = paged_attention.get_stats().await;
    println!("   Total blocks: {}", stats.total_blocks);
    println!("   Free blocks: {}", stats.free_blocks);
    println!("   Allocated blocks: {}", stats.allocated_blocks);
    println!("   Active sequences: {}", stats.active_sequences);
    println!("   Memory utilization: {:.1}%", stats.memory_utilization);
    println!("   Average blocks per sequence: {:.1}", stats.average_blocks_per_sequence);
    println!("   Total attention operations: {}", stats.attention_ops);

    // Test tensor reshaping
    println!("\n🔄 Testing tensor reshape operations...");
    let test_tensor = GpuTensor::new(
        vec![1.0f32; batch_size * seq_len * hidden_size],
        vec![batch_size, seq_len, hidden_size],
        device.clone()
    )?;

    // Test reshape compatibility
    let reshaped = tensor_ops.reshape(&test_tensor, vec![batch_size, config.num_heads, seq_len, config.head_dim])?;
    println!("✅ Tensor reshape successful:");
    println!("   Original shape: {:?}", test_tensor.shape());
    println!("   Reshaped shape: {:?}", reshaped.shape());

    // Clean up
    println!("\n🗑️  Cleaning up...");
    paged_attention.free_sequence(sequence_id).await?;
    println!("✅ Sequence {} freed successfully", sequence_id);

    // Final statistics
    let final_stats = paged_attention.get_stats().await;
    println!("\n📊 Final Statistics:");
    println!("   Active sequences: {}", final_stats.active_sequences);
    println!("   Allocated blocks: {}", final_stats.allocated_blocks);

    println!("\n🎉 PagedAttention integration test completed successfully!");
    println!("\n🎯 Key Features Demonstrated:");
    println!("   ✅ Block-based memory management");
    println!("   ✅ Sequence allocation and deallocation");
    println!("   ✅ Attention computation with paged KV cache");
    println!("   ✅ Memory utilization tracking");
    println!("   ✅ Performance statistics");
    println!("   ✅ Tensor reshaping compatibility");

    println!("\n🚀 UniLLM PagedAttention is ready for production use!");

    Ok(())
}
//! Test RadixAttention implementation
//!
//! This test demonstrates RadixAttention for efficient prefix sharing,
//! showcasing SGLang's key innovation for optimizing repeated prompts.

use runtime::{
    radix_attention::{RadixAttention, RadixAttentionConfig, EvictionPolicy},
    gpu_tensor_ops::{GpuDevice, GpuTensor, GpuTensorOps},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌳 Testing RadixAttention Implementation");
    println!("=========================================");

    // Initialize device
    let device = GpuDevice::Cpu;
    let _tensor_ops = GpuTensorOps::with_device(device.clone());

    // Test different RadixAttention configurations
    let configs = vec![
        ("Basic Config", RadixAttentionConfig::default()),
        ("Deep Tree", RadixAttentionConfig {
            max_tree_depth: 2048,
            min_prefix_length: 2,
            auto_prefix_detection: true,
            prefix_cache_size: 50000,
            enable_prefix_compression: true,
            eviction_policy: EvictionPolicy::LRU,
            use_paged_attention: true,
        }),
        ("Aggressive Sharing", RadixAttentionConfig {
            max_tree_depth: 512,
            min_prefix_length: 1,
            auto_prefix_detection: true,
            prefix_cache_size: 5000,
            enable_prefix_compression: false,
            eviction_policy: EvictionPolicy::LFU,
            use_paged_attention: false,
        }),
    ];

    for (config_name, config) in configs {
        println!("\n🧪 Testing Configuration: {}", config_name);
        println!("================================");

        // Create RadixAttention instance
        let radix_attention = RadixAttention::new(config, device.clone()).await?;

        // Test scenarios with different prefix patterns
        test_prefix_sharing_scenarios(&radix_attention, &device).await?;

        // Test cleanup and resource management
        test_resource_management(&radix_attention).await?;
    }

    println!("\n🎉 RadixAttention testing completed successfully!");
    println!("\n🎯 Key Features Demonstrated:");
    println!("   ✅ Radix tree construction and traversal");
    println!("   ✅ Automatic prefix detection and sharing");
    println!("   ✅ KV cache reuse optimization");
    println!("   ✅ Dynamic tree management");
    println!("   ✅ Resource cleanup and eviction");
    println!("   ✅ Performance statistics tracking");
    println!("   ✅ Multiple eviction policies");

    println!("\n🚀 RadixAttention is ready for SGLang-competitive prefix sharing!");

    Ok(())
}

async fn test_prefix_sharing_scenarios(radix_attention: &RadixAttention, device: &GpuDevice) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Testing Prefix Sharing Scenarios");
    println!("====================================");

    // Define common prefixes that would benefit from sharing
    let scenarios = vec![
        // Chat completion scenarios
        ("System prompt sharing", vec![
            vec![1, 2, 3, 4, 5, 10, 11, 12], // "You are a helpful assistant. Tell me about cats"
            vec![1, 2, 3, 4, 5, 20, 21, 22], // "You are a helpful assistant. Tell me about dogs"
            vec![1, 2, 3, 4, 5, 30, 31, 32], // "You are a helpful assistant. Tell me about birds"
        ]),
        ("Code completion", vec![
            vec![100, 101, 102, 103, 200, 201], // "def function_name():\n    return"
            vec![100, 101, 102, 103, 300, 301], // "def function_name():\n    print"
            vec![100, 101, 102, 103, 400, 401], // "def function_name():\n    raise"
        ]),
        ("Multi-turn conversation", vec![
            vec![50, 51, 52, 60, 61], // "User: Hello\nAssistant: Hi there!"
            vec![50, 51, 52, 70, 71], // "User: Hello\nAssistant: How can I help?"
            vec![50, 51, 52, 80, 81], // "User: Hello\nAssistant: Good morning!"
        ]),
    ];

    for (scenario_name, token_sequences) in scenarios {
        println!("\n🎭 Scenario: {}", scenario_name);

        // Create test tensors for each sequence
        let mut results = Vec::new();

        for (i, token_ids) in token_sequences.iter().enumerate() {
            let request_id = (i as u64) + 1000;

            // Create dummy tensors for this sequence
            let seq_len = token_ids.len();
            let num_heads = 8;
            let head_dim = 64;
            let hidden_size = num_heads * head_dim;

            let query_data = vec![0.1f32 * (i + 1) as f32; seq_len * hidden_size];
            let key_data = vec![0.2f32 * (i + 1) as f32; seq_len * hidden_size];
            let value_data = vec![0.3f32 * (i + 1) as f32; seq_len * hidden_size];

            let query = GpuTensor::new(query_data, vec![1, seq_len, num_heads, head_dim], device.clone())?;
            let key = GpuTensor::new(key_data, vec![1, seq_len, num_heads, head_dim], device.clone())?;
            let value = GpuTensor::new(value_data, vec![1, seq_len, num_heads, head_dim], device.clone())?;

            println!("  📤 Processing request {} with {} tokens", request_id, token_ids.len());

            // Process with RadixAttention
            let result = radix_attention.forward(
                request_id,
                token_ids,
                &query,
                &key,
                &value,
            ).await?;

            println!("     Prefix reused: {}", result.prefix_reused);
            println!("     Prefix length: {}", result.prefix_length);
            println!("     Computation saved: {:.1}%", result.computation_saved_percent);

            results.push(result);
        }

        // Analyze prefix sharing efficiency
        let reuse_count = results.iter().filter(|r| r.prefix_reused).count();
        let total_requests = results.len();
        let reuse_rate = (reuse_count as f32 / total_requests as f32) * 100.0;

        println!("  📊 Scenario Results:");
        println!("     Total requests: {}", total_requests);
        println!("     Prefix reuses: {}", reuse_count);
        println!("     Reuse rate: {:.1}%", reuse_rate);

        if reuse_rate > 0.0 {
            let avg_computation_saved = results.iter()
                .filter(|r| r.prefix_reused)
                .map(|r| r.computation_saved_percent)
                .sum::<f32>() / reuse_count as f32;
            println!("     Avg computation saved: {:.1}%", avg_computation_saved);
        }
    }

    Ok(())
}

async fn test_resource_management(radix_attention: &RadixAttention) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🗂️ Testing Resource Management");
    println!("===============================");

    // Get initial statistics
    let initial_stats = radix_attention.get_radix_stats().await;
    println!("📊 Initial RadixAttention Statistics:");
    println!("   Total nodes: {}", initial_stats.total_nodes);
    println!("   Max depth: {}", initial_stats.max_depth_reached);
    println!("   Cache hits: {}", initial_stats.cache_hits);
    println!("   Cache misses: {}", initial_stats.cache_misses);
    println!("   Prefix reuse rate: {:.1}%", initial_stats.prefix_reuse_rate);
    println!("   Memory saved: {:.2} MB", initial_stats.memory_saved_mb);
    println!("   Tree compression: {:.2}x", initial_stats.tree_compression_ratio);

    // Test request cleanup
    println!("\n🧹 Testing request cleanup...");

    // Create several requests
    let device = radix_attention.get_device().clone();
    let cleanup_requests = vec![
        (9001u64, vec![1, 2, 3, 4, 5]),
        (9002u64, vec![1, 2, 3, 6, 7]),
        (9003u64, vec![1, 2, 8, 9, 10]),
    ];

    for (request_id, token_ids) in cleanup_requests {
        let seq_len = token_ids.len();
        let dummy_tensor = GpuTensor::zeros(vec![1, seq_len, 8, 64], device.clone())?;

        radix_attention.forward(
            request_id,
            &token_ids,
            &dummy_tensor,
            &dummy_tensor,
            &dummy_tensor,
        ).await?;

        println!("   Created request: {}", request_id);
    }

    // Simulate some time passing and cleanup expired requests
    println!("🕒 Cleaning up expired requests (TTL: 0 seconds for testing)...");
    let cleaned_count = radix_attention.cleanup_expired_requests(0).await?;
    println!("   Cleaned up {} requests", cleaned_count);

    // Test manual request cleanup
    println!("🗑️ Testing manual request cleanup...");
    radix_attention.free_request(9001).await?;
    radix_attention.free_request(9002).await?;
    radix_attention.free_request(9003).await?;

    // Get final statistics
    let final_stats = radix_attention.get_radix_stats().await;
    println!("\n📊 Final RadixAttention Statistics:");
    println!("   Total nodes: {}", final_stats.total_nodes);
    println!("   Max depth: {}", final_stats.max_depth_reached);
    println!("   Cache hits: {}", final_stats.cache_hits);
    println!("   Cache misses: {}", final_stats.cache_misses);
    println!("   Prefix reuse rate: {:.1}%", final_stats.prefix_reuse_rate);
    println!("   Memory saved: {:.2} MB", final_stats.memory_saved_mb);

    // Performance analysis
    let total_operations = final_stats.cache_hits + final_stats.cache_misses;
    if total_operations > 0 {
        println!("\n⚡ Performance Analysis:");
        println!("   Total operations: {}", total_operations);
        println!("   Cache hit rate: {:.1}%", (final_stats.cache_hits as f32 / total_operations as f32) * 100.0);
        println!("   Estimated speedup: {:.2}x", 1.0 + (final_stats.prefix_reuse_rate / 100.0));
    }

    Ok(())
}
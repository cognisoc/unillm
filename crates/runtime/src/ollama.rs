//! Ollama Registry Client
//!
//! Downloads models from the Ollama registry (registry.ollama.ai).
//! Models are cached locally in ~/.cache/unillm/ollama/

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";
const DEFAULT_TAG: &str = "latest";

/// Ollama registry client for downloading models
pub struct OllamaRegistry {
    client: Client,
    cache_dir: PathBuf,
}

/// Ollama manifest format (Docker registry v2 compatible)
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub schema_version: i32,
    pub media_type: String,
    pub config: LayerInfo,
    pub layers: Vec<LayerInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LayerInfo {
    pub digest: String,
    pub size: u64,
    #[serde(rename = "mediaType")]
    pub media_type: String,
}

/// Model info extracted from manifest
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub tag: String,
    pub digest: String,
    pub size: u64,
}

impl OllamaRegistry {
    /// Create a new Ollama registry client with default cache directory
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("unillm")
            .join("ollama");

        Self::with_cache_dir(cache_dir)
    }

    /// Create a new Ollama registry client with custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;
        std::fs::create_dir_all(cache_dir.join("manifests"))?;
        std::fs::create_dir_all(cache_dir.join("blobs"))?;

        Ok(Self {
            client: Client::new(),
            cache_dir,
        })
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Parse model string into name and tag
    /// Examples: "tinyllama" -> ("tinyllama", "latest")
    ///           "qwen2.5:0.5b" -> ("qwen2.5", "0.5b")
    fn parse_model_string(model: &str) -> (&str, &str) {
        if let Some((name, tag)) = model.split_once(':') {
            (name, tag)
        } else {
            (model, DEFAULT_TAG)
        }
    }

    /// Fetch manifest for a model
    pub async fn get_manifest(&self, model: &str) -> Result<Manifest> {
        let (name, tag) = Self::parse_model_string(model);
        let url = format!("{}/v2/library/{}/manifests/{}", OLLAMA_REGISTRY, name, tag);

        println!("Fetching manifest from: {}", url);

        let response = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.docker.distribution.manifest.v2+json")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to fetch manifest: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            ));
        }

        let manifest: Manifest = response.json().await?;
        Ok(manifest)
    }

    /// Get model info from manifest
    pub async fn get_model_info(&self, model: &str) -> Result<ModelInfo> {
        let (name, tag) = Self::parse_model_string(model);
        let manifest = self.get_manifest(model).await?;

        // Find the model layer (GGUF file)
        let model_layer = manifest
            .layers
            .iter()
            .find(|l| l.media_type.contains("model"))
            .ok_or_else(|| anyhow!("No model layer found in manifest"))?;

        Ok(ModelInfo {
            name: name.to_string(),
            tag: tag.to_string(),
            digest: model_layer.digest.clone(),
            size: model_layer.size,
        })
    }

    /// Check if a model is already cached
    pub fn is_cached(&self, model: &str) -> bool {
        if let Ok(info) = tokio::runtime::Handle::current().block_on(self.get_model_info(model)) {
            let blob_path = self.blob_path(&info.digest);
            blob_path.exists()
        } else {
            false
        }
    }

    /// Get path where a blob would be cached
    fn blob_path(&self, digest: &str) -> PathBuf {
        // digest is like "sha256:abc123..."
        let filename = digest.replace(':', "_") + ".gguf";
        self.cache_dir.join("blobs").join(filename)
    }

    /// Pull (download) a model from the registry
    pub async fn pull(&self, model: &str) -> Result<PathBuf> {
        let info = self.get_model_info(model).await?;
        let blob_path = self.blob_path(&info.digest);

        // Check if already cached
        if blob_path.exists() {
            println!("Model already cached at: {}", blob_path.display());
            return Ok(blob_path);
        }

        println!(
            "Downloading {} ({:.2} MB)...",
            model,
            info.size as f64 / 1_000_000.0
        );

        // Download the blob
        let (name, _tag) = Self::parse_model_string(model);
        let url = format!(
            "{}/v2/library/{}/blobs/{}",
            OLLAMA_REGISTRY, name, info.digest
        );

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to download blob: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            ));
        }

        // Stream to file with progress
        let total_size = info.size;
        let mut downloaded: u64 = 0;
        let mut file = tokio::fs::File::create(&blob_path).await?;

        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            // Print progress every 10MB
            if downloaded % (10 * 1024 * 1024) < chunk.len() as u64 {
                println!(
                    "  Progress: {:.1}%",
                    (downloaded as f64 / total_size as f64) * 100.0
                );
            }
        }

        file.flush().await?;
        println!("Download complete: {}", blob_path.display());

        Ok(blob_path)
    }

    /// List all cached models
    pub fn list_cached(&self) -> Vec<String> {
        let blobs_dir = self.cache_dir.join("blobs");
        let mut models = Vec::new();

        if let Ok(entries) = std::fs::read_dir(blobs_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".gguf") {
                        models.push(name.to_string());
                    }
                }
            }
        }

        models
    }

    /// Delete a cached model
    pub fn delete_cached(&self, digest: &str) -> Result<()> {
        let blob_path = self.blob_path(digest);
        if blob_path.exists() {
            std::fs::remove_file(blob_path)?;
        }
        Ok(())
    }

    /// Get the path to a cached model (if it exists)
    pub fn get_cached_path(&self, model: &str) -> Option<PathBuf> {
        if let Ok(info) = tokio::runtime::Handle::current().block_on(self.get_model_info(model)) {
            let blob_path = self.blob_path(&info.digest);
            if blob_path.exists() {
                return Some(blob_path);
            }
        }
        None
    }
}

impl Default for OllamaRegistry {
    fn default() -> Self {
        Self::new().expect("Failed to create OllamaRegistry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_string() {
        assert_eq!(
            OllamaRegistry::parse_model_string("tinyllama"),
            ("tinyllama", "latest")
        );
        assert_eq!(
            OllamaRegistry::parse_model_string("qwen2.5:0.5b"),
            ("qwen2.5", "0.5b")
        );
        assert_eq!(
            OllamaRegistry::parse_model_string("llama3.2:1b"),
            ("llama3.2", "1b")
        );
    }

    #[tokio::test]
    async fn test_get_manifest() {
        // This test requires network access
        let registry = OllamaRegistry::new().unwrap();

        // Try to get manifest for a small model
        match registry.get_manifest("qwen2.5:0.5b").await {
            Ok(manifest) => {
                assert_eq!(manifest.schema_version, 2);
                assert!(!manifest.layers.is_empty());
                println!("Manifest layers: {:?}", manifest.layers.len());
            }
            Err(e) => {
                // Network errors are ok in CI
                println!("Skipping manifest test (network error): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_get_model_info() {
        let registry = OllamaRegistry::new().unwrap();

        match registry.get_model_info("qwen2.5:0.5b").await {
            Ok(info) => {
                assert_eq!(info.name, "qwen2.5");
                assert_eq!(info.tag, "0.5b");
                assert!(info.digest.starts_with("sha256:"));
                println!("Model size: {} bytes", info.size);
            }
            Err(e) => {
                println!("Skipping model info test (network error): {}", e);
            }
        }
    }
}

//! HuggingFace Hub Integration
//!
//! This module provides seamless integration with HuggingFace Hub for
//! automatic model discovery, downloading, and caching. Supports both
//! public and private repositories with authentication.

use crate::{
    safetensors_loader::{SafeTensorsLoader, SafeTensorsConfig, LoadedModel},
    gpu_tensor_ops::GpuDevice,
    types::ModelError,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    fs,
};
use serde::{Deserialize, Serialize};
use reqwest::{Client, header::HeaderMap};
use tokio::fs::{File, create_dir_all};
use tokio::io::AsyncWriteExt;

/// HuggingFace Hub client for model operations
pub struct HuggingFaceHub {
    /// HTTP client for API requests
    client: Client,
    /// Authentication token (optional)
    auth_token: Option<String>,
    /// Base URL for HuggingFace Hub
    base_url: String,
    /// Local cache directory
    cache_dir: PathBuf,
}

/// Model repository information from HuggingFace Hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRepository {
    /// Repository name (e.g., "microsoft/DialoGPT-medium")
    pub repo_id: String,
    /// Model type/architecture
    pub model_type: Option<String>,
    /// Available files in the repository
    pub files: Vec<RepositoryFile>,
    /// Model card metadata
    pub metadata: Option<ModelMetadata>,
    /// Whether the model is private
    pub is_private: bool,
}

/// File information from repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryFile {
    /// File name
    pub name: String,
    /// File size in bytes
    pub size: u64,
    /// File type (safetensors, json, etc.)
    pub file_type: String,
    /// Download URL
    pub download_url: String,
    /// SHA256 hash for verification
    pub sha256: Option<String>,
}

/// Model metadata from model card
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Model architecture
    pub architecture: Option<String>,
    /// Model tags
    pub tags: Vec<String>,
    /// License information
    pub license: Option<String>,
    /// Language support
    pub language: Vec<String>,
}

/// Download configuration
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Use resume downloads for large files
    pub resume_downloads: bool,
    /// Verify file checksums after download
    pub verify_checksums: bool,
    /// Maximum concurrent downloads
    pub max_concurrent_downloads: usize,
    /// Timeout for downloads (seconds)
    pub download_timeout_secs: u64,
    /// Only download specific file patterns
    pub file_patterns: Option<Vec<String>>,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            resume_downloads: true,
            verify_checksums: true,
            max_concurrent_downloads: 4,
            download_timeout_secs: 3600, // 1 hour
            file_patterns: None,
        }
    }
}

impl HuggingFaceHub {
    /// Create a new HuggingFace Hub client
    pub fn new<P: Into<PathBuf>>(cache_dir: P, auth_token: Option<String>) -> Result<Self, ModelError> {
        let cache_dir = cache_dir.into();

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| ModelError::LoadingError(format!("Cannot create cache directory: {}", e)))?;
        }

        // Setup HTTP client with appropriate headers
        let mut headers = HeaderMap::new();
        headers.insert("User-Agent", "UniLLM/1.0".parse().unwrap());

        if let Some(ref token) = auth_token {
            headers.insert("Authorization", format!("Bearer {}", token).parse().unwrap());
        }

        let client = Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ModelError::LoadingError(format!("Cannot create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            auth_token,
            base_url: "https://huggingface.co".to_string(),
            cache_dir,
        })
    }

    /// Download and load a model from HuggingFace Hub
    pub async fn load_model(
        &self,
        repo_id: &str,
        device: GpuDevice,
        download_config: DownloadConfig,
        safetensors_config: SafeTensorsConfig,
    ) -> Result<LoadedModel, ModelError> {
        println!("🤗 Loading model from HuggingFace Hub: {}", repo_id);

        // Step 1: Get repository information
        let repo_info = self.get_repository_info(repo_id).await?;
        println!("   📊 Repository info retrieved: {} files", repo_info.files.len());

        // Step 2: Download model files
        let local_path = self.download_model_files(&repo_info, &download_config).await?;
        println!("   📥 Model files downloaded to: {}", local_path.display());

        // Step 3: Load model using SafeTensors loader
        let loader = SafeTensorsLoader::new(device, safetensors_config);
        let model = loader.load_model(&local_path, SafeTensorsConfig::default()).await?;

        println!("   ✅ Model loaded successfully from HuggingFace Hub");

        Ok(model)
    }

    /// Get repository information from HuggingFace Hub API
    async fn get_repository_info(&self, repo_id: &str) -> Result<ModelRepository, ModelError> {
        let api_url = format!("{}/api/models/{}", self.base_url, repo_id);

        let response = self.client.get(&api_url)
            .send()
            .await
            .map_err(|e| ModelError::LoadingError(format!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ModelError::LoadingError(
                format!("API request failed with status: {}", response.status())
            ));
        }

        let api_data: serde_json::Value = response.json()
            .await
            .map_err(|e| ModelError::LoadingError(format!("Invalid API response: {}", e)))?;

        // Parse repository information
        let files = self.parse_repository_files(&api_data, repo_id).await?;
        let metadata = self.parse_model_metadata(&api_data);

        Ok(ModelRepository {
            repo_id: repo_id.to_string(),
            model_type: api_data.get("pipeline_tag").and_then(|v| v.as_str()).map(|s| s.to_string()),
            files,
            metadata,
            is_private: api_data.get("private").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }

    /// Parse file information from API response
    async fn parse_repository_files(
        &self,
        api_data: &serde_json::Value,
        repo_id: &str,
    ) -> Result<Vec<RepositoryFile>, ModelError> {
        // Get file list from repository API
        let files_url = format!("{}/api/models/{}/tree/main", self.base_url, repo_id);

        let response = self.client.get(&files_url)
            .send()
            .await
            .map_err(|e| ModelError::LoadingError(format!("Files API request failed: {}", e)))?;

        let files_data: Vec<serde_json::Value> = response.json()
            .await
            .map_err(|e| ModelError::LoadingError(format!("Invalid files API response: {}", e)))?;

        let mut files = Vec::new();

        for file_data in files_data {
            if let Some(file_name) = file_data.get("path").and_then(|v| v.as_str()) {
                let file_size = file_data.get("size").and_then(|v| v.as_u64()).unwrap_or(0);

                let file_type = if file_name.ends_with(".safetensors") {
                    "safetensors"
                } else if file_name.ends_with(".json") {
                    "json"
                } else if file_name.ends_with(".txt") {
                    "text"
                } else {
                    "other"
                }.to_string();

                let download_url = format!("{}/{}/resolve/main/{}", self.base_url, repo_id, file_name);

                files.push(RepositoryFile {
                    name: file_name.to_string(),
                    size: file_size,
                    file_type,
                    download_url,
                    sha256: None, // Would be populated from API if available
                });
            }
        }

        Ok(files)
    }

    /// Parse model metadata from API response
    fn parse_model_metadata(&self, api_data: &serde_json::Value) -> Option<ModelMetadata> {
        Some(ModelMetadata {
            name: api_data.get("modelId").and_then(|v| v.as_str()).map(|s| s.to_string()),
            description: api_data.get("description").and_then(|v| v.as_str()).map(|s| s.to_string()),
            architecture: api_data.get("library_name").and_then(|v| v.as_str()).map(|s| s.to_string()),
            tags: api_data.get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
                .unwrap_or_default(),
            license: api_data.get("license").and_then(|v| v.as_str()).map(|s| s.to_string()),
            language: api_data.get("language")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
                .unwrap_or_default(),
        })
    }

    /// Download model files to local cache
    async fn download_model_files(
        &self,
        repo_info: &ModelRepository,
        config: &DownloadConfig,
    ) -> Result<PathBuf, ModelError> {
        let model_cache_dir = self.cache_dir.join(&repo_info.repo_id.replace("/", "--"));
        create_dir_all(&model_cache_dir).await
            .map_err(|e| ModelError::LoadingError(format!("Cannot create model cache directory: {}", e)))?;

        // Filter files based on patterns
        let files_to_download: Vec<&RepositoryFile> = repo_info.files.iter()
            .filter(|file| self.should_download_file(file, config))
            .collect();

        println!("   📥 Downloading {} files...", files_to_download.len());

        // Download files with concurrency control
        let semaphore = tokio::sync::Semaphore::new(config.max_concurrent_downloads);

        let download_tasks: Vec<_> = files_to_download.into_iter().map(|file| {
            let semaphore = &semaphore;
            let model_cache_dir = &model_cache_dir;
            let client = &self.client;

            async move {
                let _permit = semaphore.acquire().await.unwrap();
                self.download_file(file, model_cache_dir, client, config).await
            }
        }).collect();

        // Wait for all downloads to complete
        let results = futures::future::join_all(download_tasks).await;

        // Check for download errors
        for result in results {
            result?;
        }

        println!("   ✅ All files downloaded successfully");

        Ok(model_cache_dir)
    }

    /// Check if a file should be downloaded based on configuration
    fn should_download_file(&self, file: &RepositoryFile, config: &DownloadConfig) -> bool {
        // Always download essential files
        if file.name == "config.json" ||
           file.name == "tokenizer.json" ||
           file.name == "tokenizer_config.json" ||
           file.file_type == "safetensors" {
            return true;
        }

        // Check file patterns if specified
        if let Some(ref patterns) = config.file_patterns {
            return patterns.iter().any(|pattern| file.name.contains(pattern));
        }

        // Skip non-essential files by default
        !matches!(file.file_type.as_str(), "other")
    }

    /// Download a single file
    async fn download_file(
        &self,
        file: &RepositoryFile,
        cache_dir: &Path,
        client: &Client,
        config: &DownloadConfig,
    ) -> Result<(), ModelError> {
        let file_path = cache_dir.join(&file.name);

        // Check if file already exists and is complete
        if file_path.exists() {
            let existing_size = tokio::fs::metadata(&file_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);

            if existing_size == file.size {
                println!("      ✓ {} (cached)", file.name);
                return Ok(());
            } else if !config.resume_downloads {
                tokio::fs::remove_file(&file_path).await.ok();
            }
        }

        println!("      ⬇️  {} ({:.2} MB)", file.name, file.size as f64 / 1e6);

        // Setup request with timeout
        let response = client
            .get(&file.download_url)
            .timeout(std::time::Duration::from_secs(config.download_timeout_secs))
            .send()
            .await
            .map_err(|e| ModelError::LoadingError(format!("Download request failed for {}: {}", file.name, e)))?;

        if !response.status().is_success() {
            return Err(ModelError::LoadingError(
                format!("Download failed for {} with status: {}", file.name, response.status())
            ));
        }

        // Create parent directories if they don't exist
        if let Some(parent) = file_path.parent() {
            create_dir_all(parent).await
                .map_err(|e| ModelError::LoadingError(format!("Cannot create directory for {}: {}", file.name, e)))?;
        }

        // Write file content
        let content = response.bytes().await
            .map_err(|e| ModelError::LoadingError(format!("Failed to read content for {}: {}", file.name, e)))?;

        let mut output_file = File::create(&file_path).await
            .map_err(|e| ModelError::LoadingError(format!("Cannot create file {}: {}", file.name, e)))?;

        output_file.write_all(&content).await
            .map_err(|e| ModelError::LoadingError(format!("Cannot write file {}: {}", file.name, e)))?;

        // Verify checksum if available
        if config.verify_checksums && file.sha256.is_some() {
            self.verify_file_checksum(&file_path, file.sha256.as_ref().unwrap()).await?;
        }

        Ok(())
    }

    /// Verify file checksum
    async fn verify_file_checksum(&self, file_path: &Path, expected_sha256: &str) -> Result<(), ModelError> {
        use sha2::{Sha256, Digest};

        let file_content = tokio::fs::read(file_path).await
            .map_err(|e| ModelError::LoadingError(format!("Cannot read file for checksum verification: {}", e)))?;

        let mut hasher = Sha256::new();
        hasher.update(&file_content);
        let computed_hash = format!("{:x}", hasher.finalize());

        if computed_hash != expected_sha256 {
            return Err(ModelError::LoadingError(
                format!("Checksum mismatch for {}: expected {}, got {}",
                        file_path.display(), expected_sha256, computed_hash)
            ));
        }

        Ok(())
    }

    /// List available models from HuggingFace Hub
    pub async fn search_models(&self, query: &str, limit: usize) -> Result<Vec<ModelRepository>, ModelError> {
        let search_url = format!("{}/api/models?search={}&limit={}", self.base_url, query, limit);

        let response = self.client.get(&search_url)
            .send()
            .await
            .map_err(|e| ModelError::LoadingError(format!("Search request failed: {}", e)))?;

        let search_results: Vec<serde_json::Value> = response.json()
            .await
            .map_err(|e| ModelError::LoadingError(format!("Invalid search response: {}", e)))?;

        let mut models = Vec::new();

        for result in search_results {
            if let Some(model_id) = result.get("modelId").and_then(|v| v.as_str()) {
                // Create minimal repository info for search results
                models.push(ModelRepository {
                    repo_id: model_id.to_string(),
                    model_type: result.get("pipeline_tag").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    files: Vec::new(), // Would be populated on demand
                    metadata: self.parse_model_metadata(&result),
                    is_private: result.get("private").and_then(|v| v.as_bool()).unwrap_or(false),
                });
            }
        }

        Ok(models)
    }

    /// Clear cache for a specific model
    pub async fn clear_model_cache(&self, repo_id: &str) -> Result<(), ModelError> {
        let model_cache_dir = self.cache_dir.join(&repo_id.replace("/", "--"));

        if model_cache_dir.exists() {
            tokio::fs::remove_dir_all(&model_cache_dir).await
                .map_err(|e| ModelError::LoadingError(format!("Cannot clear cache for {}: {}", repo_id, e)))?;

            println!("🗑️  Cleared cache for {}", repo_id);
        }

        Ok(())
    }

    /// Get cache size for all models
    pub async fn get_cache_size(&self) -> Result<u64, ModelError> {
        let mut total_size = 0u64;

        if self.cache_dir.exists() {
            let mut entries = tokio::fs::read_dir(&self.cache_dir).await
                .map_err(|e| ModelError::LoadingError(format!("Cannot read cache directory: {}", e)))?;

            while let Some(entry) = entries.next_entry().await
                .map_err(|e| ModelError::LoadingError(format!("Cannot read cache entry: {}", e)))? {

                if entry.file_type().await.map(|ft| ft.is_dir()).unwrap_or(false) {
                    total_size += self.get_directory_size(&entry.path()).await?;
                }
            }
        }

        Ok(total_size)
    }

    /// Get size of a directory recursively
    fn get_directory_size<'a>(&'a self, dir_path: &'a Path) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64, ModelError>> + 'a>> {
        Box::pin(async move {
        let mut total_size = 0u64;

        let mut entries = tokio::fs::read_dir(dir_path).await
            .map_err(|e| ModelError::LoadingError(format!("Cannot read directory {}: {}", dir_path.display(), e)))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| ModelError::LoadingError(format!("Cannot read directory entry: {}", e)))? {

            let metadata = entry.metadata().await
                .map_err(|e| ModelError::LoadingError(format!("Cannot read file metadata: {}", e)))?;

            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += self.get_directory_size(&entry.path()).await?;
            }
        }

        Ok(total_size)
        })
    }
}

/// Convenience function to load a model from HuggingFace Hub with defaults
pub async fn load_model_from_hub(
    repo_id: &str,
    device: GpuDevice,
    auth_token: Option<String>,
) -> Result<LoadedModel, ModelError> {
    // Use default cache directory
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::env::temp_dir())
        .join("unillm")
        .join("hub");

    let hub = HuggingFaceHub::new(cache_dir, auth_token)?;

    hub.load_model(
        repo_id,
        device,
        DownloadConfig::default(),
        SafeTensorsConfig::default(),
    ).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_huggingface_hub_creation() {
        let temp_dir = TempDir::new().unwrap();
        let hub = HuggingFaceHub::new(temp_dir.path(), None).unwrap();

        assert!(hub.cache_dir.exists());
    }

    #[tokio::test]
    async fn test_file_pattern_filtering() {
        let temp_dir = TempDir::new().unwrap();
        let hub = HuggingFaceHub::new(temp_dir.path(), None).unwrap();

        let config = DownloadConfig {
            file_patterns: Some(vec!["*.safetensors".to_string()]),
            ..Default::default()
        };

        let file = RepositoryFile {
            name: "model.safetensors".to_string(),
            size: 1000,
            file_type: "safetensors".to_string(),
            download_url: "".to_string(),
            sha256: None,
        };

        assert!(hub.should_download_file(&file, &config));
    }
}
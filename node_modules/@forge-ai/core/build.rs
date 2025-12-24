//! Build script for forge-core
//! 
//! Configures native bindings and platform-specific optimizations.

fn main() {
    // Build N-API module
    napi_build::setup();

    // Enable SIMD optimizations on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        println!("cargo:rustc-cfg=target_feature=\"sse4.1\"");
        
        // Check for AVX2 support (optional, for local builds)
        if std::env::var("CARGO_CFG_TARGET_FEATURE")
            .map(|f| f.contains("avx2"))
            .unwrap_or(false)
        {
            println!("cargo:rustc-cfg=target_feature=\"avx2\"");
        }
    }

    // Enable NEON on ARM
    #[cfg(target_arch = "aarch64")]
    {
        println!("cargo:rustc-cfg=target_feature=\"neon\"");
    }

    // Link against system libraries for GPU support
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        
        // Try to find CUDA installation
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search={}/lib64", cuda_path);
            println!("cargo:rustc-link-search={}/lib/x64", cuda_path);
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }

    // Rerun if these files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/attention.rs");
    println!("cargo:rerun-if-changed=src/optimizer.rs");
    println!("cargo:rerun-if-changed=src/backend.rs");
    println!("cargo:rerun-if-changed=build.rs");
}

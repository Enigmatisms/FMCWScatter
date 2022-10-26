extern crate cc;

fn main() {
    cc::Build::new()
        .flag("-std=c++17")
        .flag("-D __USE_SQUARE_BRACKETS_FOR_ELEMENT_ACCESS_OPERATOR")
        .file("cpp/src/chirp_generator.cc")
        .file("cpp/src/range_finder.cc")
        .flag("-lpthread")
        .flag("-fopenmp")
        .flag("-O3")
        .compile("libfmcw_helper.a");
    println!("cargo:rustc-flags=-l dylib=stdc++");
    println!("cargo:rustc-flags=-lgomp");

    println!("FMCW cpp end compilation...");

    cc::Build::new()
        .cuda(true)
        .flag("-std=c++14")
        .flag("-lcuda")
        .flag("-lcudart")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        .flag("-rdc=true")              // enable CUDA separate compilation
        .file("curt/src/extern_func.cu")
        .file("curt/src/ray_trace_host.cu")
        .file("curt/src/ray_trace_kernel.cu")
        .file("curt/src/sampler_kernel.cu")
        .file("curt/src/scatter_kernel.cu")
        .compile("librt_helper.a");
    // Maybe two Build instances solves the problem
}

extern crate cc;

fn main() {
    cc::Build::new()
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-fopenmp")
        .flag("-lpthread")
        .file("cpp/src/chirp_generator.cc")
        .file("cpp/src/range_finder.cc")
        .compile("libfmcw_helper.a");
}

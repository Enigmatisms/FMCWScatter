extern crate cc;

fn main() {
    cc::Build::new()
        .flag("-std=c++17")
        .flag("-D __USE_SQUARE_BRACKETS_FOR_ELEMENT_ACCESS_OPERATOR")
        .file("cpp/src/chirp_generator.cc")
        .file("cpp/src/range_finder.cc")
        .flag("-lpthread")
        .flag("-O3")
        .compile("libfmcw_helper.a");
    println!("cargo:rustc-flags=-l dylib=stdc++");
    println!("cargo:rustc-flags=-lgomp");
}

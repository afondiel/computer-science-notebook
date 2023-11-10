# Rust Notes

## Overview

This is a "Hello World" resource for Rust Programming Languages.

![](https://marketsplash.com/content/images/2023/09/rust-machine-learning.png)

Src: [Rust For Machine Learning: Algorithms And Implementation - Market Splash](https://marketsplash.com/tutorials/rust/rust-machine-learning/)

## Rust in a Nutshell

- Statically typed
	- everything is known at compile time
- Multi-paradigm
	- Imperative
	- Support polymorphism via traits
	- influenced by funcional programming languages
		-Ocaml, F#, Haskell
- Focus
	- Memory safety
	- Thread safety
	- Performance
- Systems programming language
	- anything can be built with Rust even Os
- R everywhere : 
	- windows, Linux, MacOS ...
	- Embedded systems ? 
- No garbage collector
- Null free (C++)

## C# vs Rust 

|C#|Rust|
|--|--|
|Uses homebrewed tool chain |Uses the LLVM toolchain|
|Compiles to CIL bytecode | - Widely used (Swift, Clang, etc) | 
|JIT (Just-In-lime) compiled  | - Synergistic effect for platform support|
| - RyuJit|Complies to LLVM IR bytecode |
| Supports AOT (Ahead-Of-Time) compilation|AOT (Ahead-Of-Time) compiled |
| Native, architecture specific code | - Native, architecture specific code |
| Usually long lead time for new platform support |- Small binaries |
| Includes the .NET runtime|- No runtime (kind of) |

## Variables - Primitive types 

|C#|Rust|
|--|--|
|bool | bool|
|sbyte, short, int, long, nint (arch)| i8, i16, i32, i64, i128, isize (arch) | 
|byte, ushort, uint, ulong, nuint (arch) | u8. u16, u32. u64, u128, usize (arch)|
| float (single precision), double (double precision) |f32 (single precision), 164 (double precision)  |
| char (utf-16 encoded)  |char (utf-8 encoded)  |
| string | String, &str |


## Functions 

A function definition starts with the keywork `fn`

```rs
fn function_name(){
	// statements;
}
```
## Applications

- gaming engines
- operating systems
- browsers that demand scalability
- Embedded systems
- ...
## Tools & Frameworks

- Cargo (packages management already included by default during the installation)
- MS Visual Studio (from 2013 - * ) 
- Raspberry Pi (embedded applications)
- Linux

## Compilation

1. Run `Cargo` :  

```sh
cargo run
``` 

2. Create a binary

```rs
rustc main.rs
```

This compiles the code and creates a binary in your current working dir `main` , then run it by executing 

```sh
./main
```

## Hello World!


```rs
fn main(){
	println!("Hello World!");
}
```


Try out yourself on [replit playground](https://replit.com/@afondiel/rust-notes#src/main.rs) and see more examples!

## Rust & Machine Learning

- [Rust ML tools](https://www.arewelearningyet.com/)
- [rustlearn - a machine learning API by @maciejkula (Maciej Kula)](https://github.com/maciejkula/rustlearn)

## Rust & Computer Vision

- [Rust Computer Vision Project](https://github.com/rust-cv)


## References

Documentation: 
- Wikipedia : https://en.wikipedia.org/wiki/Rust_(programming_language)
- [The Rust Programming Language](https://doc.rust-lang.org/book/title-page.html) 
- installation guide :  
  - https://rustup.rs/
  - https://doc.rust-lang.org/book/ch01-01-installation.html
  - https://www.rust-lang.org/tools/install
- MSVC prerequistes : https://rust-lang.github.io/rustup/installation/windows-msvc.html
- [Embedded Rust documentation](https://docs.rust-embedded.org/)

Books: 

- [Programming Rust - Jim Blandy, Jason Orendorff - 1st-edition-2017](https://github.com/afondiel/cs-books/blob/main/computer-science/programming/rust/programmingrust-1st-edition-2017-Jim%20Blandy%2C%20Jason%20Orendorff.pdf)

- [The Embedded Rust Book](https://docs.rust-embedded.org/book/index.html)


Courses & Tutorials: 

- [Rust For Undergrads - Udemy Free Course](https://www.udemy.com/course/rust-for-undergrads/)
- Rust crash courses by [MS Reactor](https://www.youtube.com/watch?v=wHDYReCysVY) - C# Applications

Projects : 
- CLI calculator :  https://www.youtube.com/watch?v=MsocPEZBd-M


> ## "Keep up your bright swords, for the dew will rust them." - William Shakespeare


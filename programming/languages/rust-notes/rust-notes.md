# Rust Notes

## What's Rust ? 

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

## Hello World !

Save the chunck of code below in a file : `main.rs`

```rs
fn main(){
	println!("Hello World!");
}
```

## References

Documentation : 
- Wikipedia : https://en.wikipedia.org/wiki/Rust_(programming_language)
- [The Rust Programming Language](https://doc.rust-lang.org/book/title-page.html) 
- installation guide :  
  - https://rustup.rs/
  - https://doc.rust-lang.org/book/ch01-01-installation.html
  - https://www.rust-lang.org/tools/install
- MSVC prerequistes : https://rust-lang.github.io/rustup/installation/windows-msvc.html
- [Embedded Rust documentation](https://docs.rust-embedded.org/)

Books : 

- [Programming Rust - Jim Blandy, Jason Orendorff - 1st-edition-2017](https://github.com/afondiel/cs-books/blob/main/computer-science/programming/rust/programmingrust-1st-edition-2017-Jim%20Blandy%2C%20Jason%20Orendorff.pdf)

- [The Embedded Rust Book](https://docs.rust-embedded.org/book/index.html)


Courses : 

- [Rust For Undergrads - Udemy Free Course](https://www.udemy.com/course/rust-for-undergrads/)
- Rust crash courses by [MS Reactor](https://www.youtube.com/watch?v=wHDYReCysVY) - C# Applications


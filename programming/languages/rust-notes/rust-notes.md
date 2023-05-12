======= Rust notes based on C# programming  ==========

What's Rust ? 
- statically typed
	- everything is known at compile time
- Multi-paradigm
	- Imperative
	- Support polymorphism via traits
	- influenced by funcional programming languages
		-Ocaml, F#, Haskell
- focus
	- Memory safety
	- Thread safety
	- Performance
- systems programming language
	- anything can be built with Rust even Os
- run everywher : 
	- windows, Linux, MacOS ...
	- Embedded systems ? 
- No garbage collector
- Null free( C++)

C# vs Rust 
- C#
• Uses homebrewed tool chain 
• Compiles to CIL bytecode 
• JIT (Just-In-lime) compiled 
	• RyuJit 
• Supports AOT (Ahead-Of-Time) compilation 
	• Native, architecture specific code 
	• Usually long lead time for new platform support 
	• Includes the .NET runtime 
Rust 
• Uses the LLVM toolchain 
	• Widely used (Swift, Clang, etc) 
	• Synergistic effect for platform support 
• Complies to LLVM IR bytecode 
• AOT (Ahead-Of-Time) compiled 
	• Native, architecture specific code 
	* Small binaries 
	• No runtime (kind of) 

Variables declarations
C# vs Rust



Primitive types 
C# vs Rust
Rust boo! boot i8,116,132,164. i128, isize (arch) sbyte, short, int, long, nint (arch) u8. ul 6, u32. u64, u128, usize (arch) byte, ushort, uint, ulong, nuint (arch) 132 (single precision), 164 (double precision) float (single precision), double (double precision) char (Off 8 encoded) char (uTF 16 encoded) String, &str string 

Functions 
C# vs Rust


# References 
Notes based on Rust crash courses by [MS Reactor](https://www.youtube.com/watch?v=wHDYReCysVY)
[Embedded Rust documentation](https://docs.rust-embedded.org/)
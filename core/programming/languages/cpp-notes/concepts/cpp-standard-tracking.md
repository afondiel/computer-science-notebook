# C++ Standard Feature Tracking

A comprehensive overview of features introduced in each C++ standard, with code examples and references.

## Table of Contents

- [C++98 (ISO/IEC 14882:1998)](#c98-isoiec-148821998)
    - [Explanation](#explanation)
- [C++11 (ISO/IEC 14882:2011)](#c11-isoiec-1488212011)
    - [Explanation](#explanation-1)
- [C++14 (ISO/IEC 14882:2014)](#c14-isoiec-1488212014)
    - [Explanation](#explanation-2)
- [C++17 (ISO/IEC 14882:2017)](#c17-isoiec-1488212017)
    - [Explanation](#explanation-3)
- [C++20 (ISO/IEC 14882:2020)](#c20-isoiec-1488212020)
    - [Explanation](#explanation-4)
- [C++23 (ISO/IEC 14882:2024)](#c23-isoiec-1488212024)
    - [Explanation](#explanation-5)
- [Future Directions](#future-directions)
- [References & Resources](#references--resources)

## C++98 (ISO/IEC 14882:1998)
| Feature              | Description                          | Code Snippet                     | ISO Reference              |
|----------------------|--------------------------------------|----------------------------------|----------------------------|
| Standard Template Library (STL) | Containers, algorithms, and iterators | `std::vector<int> v;`            | ISO/IEC 14882:1998 §17     |
| Namespaces           | Prevent naming collisions            | `namespace myns { ... }`         | ISO/IEC 14882:1998 §7.3    |
| Exceptions           | Error handling via `try`/`catch`     | `throw std::runtime_error("msg")`| ISO/IEC 14882:1998 §15     |
| RTTI (Runtime Type Information) | Enables dynamic type checking       | `typeid(x).name()`               | ISO/IEC 14882:1998 §5.2    |
| bool type            | Boolean type                         🙂 `bool b = true;`                 | ISO/IEC 14882:1998 §3.9.1  |

### Explanation
- **STL**: Provides reusable containers like `std::vector` and algorithms like `std::sort`, forming the backbone of C++’s standard library.
- **Namespaces**: Allow developers to organize code and avoid name conflicts, especially in large projects.
- **Exceptions**: Enable robust error handling, allowing programs to recover from unexpected conditions.
- **RTTI**: Supports dynamic type identification, useful in polymorphic scenarios.
- **bool type**: Introduces a dedicated boolean type, improving code clarity over using integers for boolean logic.

## C++11 (ISO/IEC 14882:2011)
| Feature              | Description                          | Code Snippet                     | ISO Reference              |
|----------------------|--------------------------------------|----------------------------------|----------------------------|
| `auto` type deduction | Compiler infers variable types       | ` Wauto x = 42;`                   | ISO/IEC 14882:2011 §7.1.6  |
| Lambda expressions   | Anonymous functions                  | `[] (int x) { return x*2; }`     | ISO/IEC 14882:2011 §5.1.2  |
| Smart pointers       | `std::unique_ptr`, `std::shared_ptr` | `std::unique_ptr<int> p(new int);` | ISO/IEC 14882:2011 §20.7  |
| Move semantics       | Efficient transfer of resources      | `std::move(x)`                   | ISO/IEC 14882:2011 §12.8   |
| Variadic templates   | Templates with variable arguments    | `template <typename... Args> void f(Args... args);` | ISO/IEC 14882:2011 §14.5.3 |

### Explanation
- **auto**: Simplifies variable declarations by letting the compiler deduce types, reducing boilerplate.
- **Lambda expressions**: Enable inline function definitions, useful for algorithms and callbacks.
- **Smart pointers**: Automate memory management, reducing memory leaks with `std::unique_ptr` and `std::shared_ptr`.
- **Move semantics**: Optimize resource transfer (e.g., strings, vectors) by avoiding copies, using rvalue references.
- **Variadic templates**: Allow templates to accept any number of arguments, enhancing generic programming.

## C++14 (ISO/IEC 14882:2014)
| Feature              | Description                          | Code Snippet                     | ISO Reference              |
|----------------------|--------------------------------------|----------------------------------|----------------------------|
| Generic lambdas      | Lambdas with `auto` parameters       | `auto f = [](auto x) { ... };`   | ISO/IEC 14882:2014 §5.1.2  |
| Heterogeneous lookup | Search associative containers with non-key types | `std::set<std::string> s; s.find("key");` | ISO/IEC 14882:2014 §23.2.4 |
| `std::shared_mutex`  | Shared/read-write locks              | `std::shared_lock lk(mutex);`    | ISO/IEC 14882:2014 §30.4   |
| decltype(auto)       | More precise type deduction          | `decltype(auto) x = y;`          | ISO/IEC 14882:2014 §7.1.6.4|
| Binary literals      | Binary integer literals              | `unsigned x = 0b1010;`           | ISO/IEC 14882:2014 §2.14.2 |

### Explanation
- **Generic lambdas**: Allow lambda parameters to use `auto`, making lambdas more flexible for generic code.
- **Heterogeneous lookup**: Enables searching containers like `std::set` with compatible types, improving performance.
- **std::shared_mutex**: Supports shared locking for read-heavy multithreaded applications.
- **decltype(auto)**: Preserves reference and cv-qualifiers in type deduction, useful for perfect forwarding.
- **Binary literals**: Improve readability for bit-level operations with `0b` prefix.

## C++17 (ISO/IEC 14882:2017)
| Feature              | Description                          | Code Snippet                     | ISO Reference              |
|----------------------|--------------------------------------|----------------------------------|----------------------------|
| Structured bindings  | Decompose tuples/structs             | `auto [x, y] = std::pair(1, 2);` | ISO/IEC 14882:2017 §11.5   |
| `if constexpr`       | Compile-time conditional code        | `if constexpr (std::is_integral_v<T>) { ... }` | ISO/IEC 14882:2017 §9.4.1 |
| `std::filesystem`    | Filesystem operations                | `std::filesystem::path p{"file.txt"};` | ISO/IEC 14882:2017 §30.10 |
| Template argument deduction | For class templates                  | `std::vector v{1,2,3};`          | ISO/IEC 14882:2017 §13.3.10|
| Fold expressions     | Simplify parameter pack operations   | `(... + args)`                   | ISO/IEC 14882:2017 §14.5.3 |

### Explanation
- **Structured bindings**: Simplify unpacking of tuples, pairs, or structs into individual variables.
- **if constexpr**: Enables compile-time branching, optimizing code generation for templates.
- **std::filesystem**: Provides portable filesystem operations, like path manipulation and file I/O.
- **Template argument deduction**: Allows class templates (e.g., `std::vector`) to deduce types from initializers.
- **Fold expressions**: Streamline operations on parameter packs, making variadic templates easier to use.

## C++20 (ISO/IEC 14882:2020)
| Feature              | Description                          | Code Snippet                     | ISO Reference              |
|----------------------|--------------------------------------|----------------------------------|----------------------------|
| Concepts             | Constrain template parameters        | `template <std::integral T> ...` | ISO/IEC 14882:2020 §7.6   |
| Modules              | Replace headers with modules         | `import std.core;`               | ISO/IEC 14882:2020 §10.1  |
| Three-way comparison (`<=>`) | Simplify comparison operators | `auto operator<=>(const T&) = default;` | ISO/IEC 14882:2020 §11.10 |
| Coroutines           | Asynchronous programming             | `co_await some_async_function();` | ISO/IEC 14882:2020 §14.4  |
| Ranges               | Lazy, composable sequences           | `std::views::iota(1, 10) | std::views::filter([](int x){ return x % 2 == 0; })` | ISO/IEC 14882:2020 §23.16|

### Explanation
- **Concepts**: Restrict template parameters to specific types, improving error messages and code clarity.
- **Modules**: Replace traditional headers with a modular system, reducing compilation times.
- **Three-way comparison**: Simplifies writing comparison operators with a single `<=>` operator.
- **Coroutines**: Support asynchronous programming with `co_await`, `co_yield`, and `co_return`.
- **Ranges**: Provide a modern, composable way to process sequences, replacing some STL algorithms.

## C++23 (ISO/IEC 14882:2024)
| Feature              | Description                          | Code Snippet                     | ISO Reference              |
|----------------------|--------------------------------------|----------------------------------|----------------------------|
| `std::print`         | Type-safe formatted output           | `std::print("Hello {}!", name);` | ISO/IEC 14882:2024 §30.5   |
| `std::mdspan`        | Multidimensional array views         | `std::mdspan<int, 2> mat(data);` | ISO/IEC 14882:2024 §24.7   |
| Deducing `this`      | Simplify CRTP patterns               | `void foo(this auto&& self) { ... }` | ISO/IEC 14882:2024 §12.2.2 |
| `std::expected`      | Expected value or error              | `std::expected<int, std::string> result = get_value();` | ISO/IEC 14882:2024 §26.9   |

### Explanation
- **std::print**: Offers type-safe, efficient formatted output, improving over `printf`.
- **std::mdspan**: Provides views for multidimensional arrays, useful in scientific computing.
- **Deducing this**: Simplifies patterns like CRTP by making the `this` pointer’s type deducible.
- **std::expected**: Enhances error handling by storing either an expected value or an error.


## Future Directions
C++26 is under development, with potential features like reflection and metaprogramming. These could further enhance C++’s capabilities, continuing its evolution as a leading systems programming language.


## References & Resources
- [C++ Standard Documents (ISO)](https://www.iso.org/standard/79358.html)  
- [C++ Reference (cppreference)](https://en.cppreference.com)  
- [GCC C++ Standards Support](https://gcc.gnu.org/projects/cxx-status.html)  
- [Modern C++ Features (GitHub)](https://github.com/AnthonyCalandra/modern-cpp-features)  
- [Microsoft C++ Docs](https://learn.microsoft.com/en-us/cpp/cpp/)  
- [C++ Wikipedia](https://en.wikipedia.org/wiki/C++)  


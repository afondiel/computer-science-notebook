
# Java Notes :coffee:


## Overview

Java is a popular programming language used to develop mobile apps, web apps, desktop apps, games and much more.

Java is an Object Orient Program (OOP) 

## Variables/attributes  

attribute(vars of type private) = atomic!!!!


## Operations/Functions 

Getter and setter : way to access and modify an attribute
        
- setBalance(): WRITE
- getBalance(): READ

##  Class and Object 

`Class` is a blueprint or template from which objects are created 

- variables/attributes + Operations/Functions

`Object` is an instance of a class at runtime. 

class: Student{}

```java
Student myStudent = new Student(); // instantiation
```

### super vs this

- `super`: method to access parent method
- `this` : current (daughter?) class method


## OOP Concepts

### Abstraction

process of displaying only required information to the user by hiding other details

### Encapsulation
- private
- audience
- protected(visible to inheritors)

### Inheritange

An object `a` (parent) can take `b`


### Polymorphism (static & dynamic) => Java Revolution !!!

Evolution and reuse of the object during its development



## Hello World ! 

```java
class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!"); 
    }
}
```


## References 

Wikipedia : 
- https://en.wikipedia.org/wiki/Java_(programming_language)

Wikibooks
- https://en.wikibooks.org/wiki/Java_Programming

W3schools Tutorial : 
- https://www.w3schools.com/java/default.asp

Tools & Frameworks

- https://www.oracle.com/java/

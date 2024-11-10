# Design Patterns Notes 


**Table of Contents**

- [Design Patterns Notes](#design-patterns-notes)
  - [Overview](#overview)
  - [Fundamentals OOP](#fundamentals-oop)
  - [Reference \& Classes](#reference--classes)
  - [UML Basics](#uml-basics)
  - [UML Diagrams](#uml-diagrams)
    - [Structural](#structural)
    - [Behavioral :](#behavioral-)
  - [Design Patterns history](#design-patterns-history)
  - [Design Patterns concept](#design-patterns-concept)
  - [Design Patterns Organization](#design-patterns-organization)
  - [Creationals patterns](#creationals-patterns)
    - [Singleton](#singleton)
    - [Factory Method (\*)](#factory-method-)
    - [Abstract Factory](#abstract-factory)
    - [Builder](#builder)
    - [Prototype](#prototype)
  - [Structurals patterns](#structurals-patterns)
    - [Adapter (i.e. wrapper)](#adapter-ie-wrapper)
    - [Bridge](#bridge)
    - [Composite](#composite)
    - [Decorator](#decorator)
    - [Facade](#facade)
    - [Flyweight](#flyweight)
    - [Proxy](#proxy)
  - [Behaviorals patterns](#behaviorals-patterns)
    - [Interpreter](#interpreter)
    - [Template method](#template-method)
    - [Chain of responsibility](#chain-of-responsibility)
    - [Command](#command)
    - [Iterator](#iterator)
    - [Mediator](#mediator)
    - [Memento](#memento)
    - [Observer](#observer)
    - [State](#state)
    - [Strategy](#strategy)
    - [Visitor](#visitor)
  - [Design Patterns - Usage](#design-patterns---usage)
  - [Identify the problem](#identify-the-problem)
  - [Find the pattern](#find-the-pattern)
  - [Apply the pattern](#apply-the-pattern)
  - [Hello World!](#hello-world)
  - [References](#references)


## Overview

This is Design Patterns "Hello World" resources

## Fundamentals OOP

1. **Encapsulation** : data + procedures/functions
2. **Message** : different from static binding (where is the compiler responsible for compilation + final link), in message passing the objects are responsible for executing the code themselves (internally) : methods and attributes (dynamic binding) in (execution time)
3. **Polymorphism** : a caller(method/function/calls w/ the same) can create/replace any object in different way
4. **Inheritance** : create an object from an existing one (parent) with the same propreties (attributs and methods)
5. **Abstraction** : concept of hiding the internal implementation details and only focus on inputs and outputs (ex: a coffee machine ..)

## Reference & Classes

- Variables :  Attribute w/ a reference (memory adress) : a name, reference, object
- Types : 
  - Class : Var + methods for future instance
  - Type a variable or a return type
  - Mold type ?   : A factory

- Interface : 
Abstract type (abstract class) w/ methodes signatures (prototypes)
=> In the class we implement one/many interfaces methods


## UML Basics

Class Model  : 

```
--------------
| className	  |
--------------
| Attributes  |	
--------------
| Operation1()|
| Operation2()|
| Operation3()|
--------------
```
Encapsulation : + (public); = protected; - private


Inheritance
```
--------------
| superClass  |	
--------------
       ^
       |
       |
       |
       |
--------------
| SubClass   |	
--------------
```

Abstract classes : If a class doesn't provide an implementation for any operation
``` 
-------------------
| AbstractClass	   |	
----------------------
| AbstractOperation()|	
----------------------
```

Instance
```
--------------
| 	Class     |	
--------------
       ^
       |
       | <<instanceOf>>
       |
       |
--------------
|	 Object   |	
--------------
```

Relation between classes

- Dependency -----> (import; return values, params )
- Association/multiple associations : both instance objects are referenced
- Agregation : strong link
- Composition : STRONGEST link

## UML Diagrams

### Structural 
- Class diagram
- Component diagram
- Composite structure diagram
- Deployment diagram
- Object diagram
- Package diagram
- Profile diagram

### Behavioral : 
- Activity diagram
- Communication diagram
- Interaction overview diagram
- Sequence diagram
- State diagram
- Timing diagram
- Use case diagram


## Design Patterns history 

Inspiration from architecture : 
by Christopher Alexander. Form the book, "A pattern language" (1977)

"FIRST" Software design patterns book : 
-  "Design Patterns: Elements of Reusable Object-Oriented Software" (1995), by Erich Gamma, Richard Helm, Ralph Johnson and John Vlissides the Gang of Four (GoF).

## Design Patterns concept 

What is NOT a pattern : 
- a reusable or parametrizable componant
- an algorithm.
- a framework.

What is a pattern : 
- The identification by a name explicite who identify a the pattern
- The description of the problem
- The problem's solution
- The positive and negative solution's concequencies

## Design Patterns Organization 

![](https://scaler.com/topics/images/types_of_patterns.webp)

The 23 originals patterns referenced at the GoF's patterns are organized into 3 big families : 

## Creationals patterns

Abstract the instanciation process.

### Singleton

### Factory Method (*)

Manufactures an object

- case 1: Creator is an abstract (super)class
- case 2: Creator is an Concrete (super)class
- case 3: Parameterized factory methods => create multiple products

- Proprieties
  - when creating an object without knowing it in davanced during the design stage 
  (ex : an random event of same nature, decision (person class : teacher/student ) ...
  - create new objects(products) during the run-time
  - subclasses(creator) create the product instead of superclass(Creator)
  - subclasses(creator) overrides methods/operations of the superclasses(Creator)

- Applications: 
  - create new objects(products) during the run-time
  - useful for large projects

### Abstract Factory
### Builder
### Prototype

## Structurals patterns

Compose the objects and classes in large structures

- It uses *inheritance* to compose interfaces or implementations  

### Adapter (i.e. wrapper)

Interface convertor

* Proprieties * 
- case 1 : create an adapter class?
- case 2 : work through the adapter to convert the incomptible class to be used by the client ?

* Applications : 
- How can a class be *reused* that does not have an interface that a client requires?
- How can classes that have incompatible interfaces work together?
- How can an alternative interface be provided for a class?
- Gateway? API ? 

### Bridge
### Composite
### Decorator 
### Facade
### Flyweight
### Proxy

## Behaviorals patterns

Assign responsiblities between the objects.

### Interpreter
### Template method 
### Chain of responsibility
### Command
### Iterator

### Mediator

Define an object that encapsulates how a set of objects interact

* Proprieties * 
- case1 : Mediator promotes loose coupling by keeping objects from referring to each other explicitly
- case2 : it lets you vary their interaction independently.
* Applications : 
- ?

### Memento
### Observer 
### State
### Strategy
### Visitor 


## Design Patterns - Usage	  

- Find the problem 
- Find the generic pattern for this problem 
- Create his own solution by reusing the generic solution 

## Identify the problem

- understand well his conception problem
/!\ get to know very all the patterns will much easier to identify the pattern that fits the best /!\

## Find the pattern

-identify the pattern(s)
-identify the paticipants
-study the impact/consequence of its use

## Apply the pattern

-Finally, to create our own solution by reusing the generic solution.
-We match our participants and pattern's ones 
-we recreate the pattern schematic with our own names

/!\ Usage : rename the class WidgetGenerator => WidgetFactory (express more clearly the pattern used)

## Hello World!

Refer to this [repo](./patterns/).


## References

- [Refactoring.guru](https://refactoring.guru/design-patterns)
- [Software Design Pattern](https://en.wikipedia.org/wiki/Software_design_pattern)
- [Codingame - design patterns - uml basics](https://www.codingame.com/playgrounds/503/design-patterns/uml-basics)

Books:

- [Computer Science - design patterns Books](https://github.com/afondiel/cs-books/tree/main/computer-science/design-patterns)

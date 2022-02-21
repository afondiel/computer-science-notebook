===================================
===== Design-patterns-notes =======
===================================


### Fundamentals OOP ####

1/ Encapsulation
2/ Message
Messaging passing execution time)
3/ Polymorphism
4/ Inheritance

# Reference & Classes

* Variables : 
Attribute w/ a reference (memory adress) : a name, reference, object
* Types : 
-> Class : 
- Var + methods for future instance
- type a variable or a return type
- mould type ?   : A factory

* Interface : 
Abstract type (abstract class) w/ methodes signatures (prototypes)
=> In the class we implement one/many interfaces methods


#### UML BASICS ####

Class Model  : 

--------------
| className	  |
--------------
| Attributes  |	
--------------
| Operation1()|
| Operation2()|
| Operation3()|
--------------
Encapsulation : + (public); = protected; - private


Inheritance
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


Abstract classes : If a class doesn't provide an implementation for any operation
-------------------
| AbstractClass	   |	
----------------------
| AbstractOperation()|	
----------------------


Instance
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


Relation between classes

-Dependency -----> (import; return values, params )
-Association/multiple associations : both instance objects are referenced
-Agregation : strong link
-Composition : STRONGEST link

## UML DRIAGRAMS ##

# Structural :
-Class diagram
-Component diagram
-Composite structure diagram
-Deployment diagram
-Object diagram
-Package diagram
-Profile diagram

# Behavioral : 
-Activity diagram
-Communication diagram
-Interaction overview diagram
-Sequence diagram
-State diagram
-Timing diagram
-Use case diagram

##################################
#### Design Patterns history ####
##################################

Inspiration from architecture : 
by Christopher Alexander. Form the book, "A pattern language" (1977)

"FIRST" Software design patterns book : 
-  "Design Patterns: Elements of Reusable Object-Oriented Software" (1995), by Erich Gamma, Richard Helm, Ralph Johnson and John Vlissides the Gang of Four (GoF).

##################################
#### Design Patterns concept ####
##################################

What is NOT a pattern : 
- a reusable or parametrizable componant.
- an algorithm.
- a framework.

What is a pattern : 
The identification by a name explicite who identify a the pattern
The description of the problem
The problem's solution
The positive and negative solution's concequencies

##################################
#### Design Patterns Organization ####
##################################

The 23 originals patterns referenced at the GoF's patterns are organized into 3 big families : 

# Creationals patterns: abstract the instanciation process

==> Singleton  <==

==> Factory Method (*) : manufactures an object <==
case 1 : Creator is an abstract (super)class
case 2 : Creator is an Concrete (super)class
case 3 : Parameterized factory methods => create multiple products

* Proprieties
- when creating an object without knowing it in davanced during the design stage 
(ex : an random event of same nature, decision (person class : teacher/student ) ...
- create new objects(products) during the run-time
- subclasses(creator) create the product instead of superclass(Creator)
- subclasses(creator) overrides methods/operations of the superclasses(Creator)

* Applications : 
- create new objects(products) during the run-time
- useful for large projects

==> Abstract Factory <==
==> Builder <==
==> Prototype <==

# Structurals patterns: compose the objects and classes in large structures

=> Usage :  
=>

==> Adaptor (i.e. wrapper) : interface convertor
* Proprieties

- case 1 :
- case 2 :
- case 3 :

* Applications : 
->
->
->

==> Bridge <==
==> Composite <==
==> Decorator <==
==> Facade <==
==> Flyweight <==
==> Proxy <==

# Behaviorals patterns: assign responsiblities between the objects

==> Interpreter <==
==> Template method <==
==> Chain of responsibility <==
==> Command <==
==> Iterator <==
==> Mediator <==
==> Memento <==
==> Observer <==
==> State <==
==> Strategy <==
==> Visitor <==

##################################
#### Design Patterns USAGE 	  ####
##################################

*** Find the problem ***
*** Find the generic pattern for this problem ***
*** Create his own solution by reusing the generic solution ****


# Identify the problem
- understand well his conception problem
/!\ get to know very all the patterns will much easier to identify the pattern that fits the best /!\

# Find the pattern

-identify the pattern(s)
-identify the paticipants
-study the impact/consequence of its use

# Apply the pattern

-Finally, to create our own solution by reusing the generic solution.
-We match our participants and pattern's ones 
-we recreate the pattern schematic with our own names

/!\ Usage : rename the class WidgetGenerator => WidgetFactory (express more clearly the pattern used)


// srcs : 
- https://en.wikipedia.org/wiki/Software_design_pattern
- https://www.codingame.com/playgrounds/503/design-patterns/uml-basics
- https://www.amazon.fr/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612


# Course Notes

The Kolmogorov complexity is the minimal length of a description of the object. To eliminate bugs early you can do test-driven development, behavior driven development, use the agile development process.

**Week 2**

The inheriance property assigns a base class to a class. The Liskov Substitution Principle, or LSP principle, says that the subclass is a specialized version of the base class, all methods of the subclass are usable through the base class interface, and the base class can be replaced by a subclass. The constructor for the Square is just written without the keywords public class in front of the name of the constructor. This constructor throws an IllegalArgumentException. Use the final keyword for methods you don't want to override. Use private if you only want the method/property accessible from within the class. If you want it to be accessible in inherited classes, use protected.
 
Use containment when classes share common data but not behavior. Inheritance is used if classes share common behavior. In Java, extends is used for extending a class and implements is used for implementing the interfaces. It’s the main difference between extends vs implements. In Java, we can inherit the fields and methods of a class by extending it using extends keyword. Please note that in Java, a class can extend maximum one class only. Interfaces are way to enforce a contract in Java. They force the implementing class to provide a certain behavior. To implement an interface, class must use implements keyword. In Java, we can implement more than one interfaces. In this case, class must implement all the methods from all the interfaces. (or declare itself abstract). The implementation is done in the class using the @Override keyword. The keyword transient in Java used to indicate that the variable should not be serialized. 

We have the following method definitions :

```
void putAll(Map<? extends K, ? extends V> m);
Set<Map.Entry<K,V>> entrySet();
```
or the StringBugger() class you can use the append(String s) method and you can also use the following method : int index = s.indexOf(String.valueOf(char ch)). You can also decide to delete the character at a specific index meaning you can run the following method : `s.deleteCharAt(index)`. The indexOf methos returns -1 if the character we are considering is not part of the StringBuffer. One way to use the assert method is by writing for example :

```
assert (condition that should be true) : "string to show when condition is not true";
```

The Java volatile keyword is used to mark a Java variable as "being stored in main memory". More precisely that means, that every read of a volatile variable will be read from the computer's main memory, and not from the CPU cache.

**Week 3**

The Javadoc is written within /** ... */. We write the parameter in @param, the return in @return and the exception throws in @throws. 

To create an array of chars you can write `char array[n]` which is of size n. Reference: A reference is a variable that refers to something else and can be used as an alias for that something else. Pointer: A pointer is a variable that stores a memory address, for the purpose of acting as an alias to what is stored at that address. So, a pointer is a reference, but a reference is not necessarily a pointer. Pointers are a particular implementation of the concept of a reference, and the term tends to be used only for languages that give you direct access to the memory address.

- C/C++ allows pointer arithmetic but Java Pointers (References) not: The term “pointer” is strongly associated with the C/C++ concept of pointers, which are variables which store memory addresses and can be modified arithmetically to point to arbitrary addresses.
In Java, pointers only exist as an implementation detail for References. A copy of the reference is copied to the stack of a called function, pointing to the same object as the calling function and allowing you to manipulate that object. However you cannot change the object the calling function refers to.
- Java doesn’t support pointer explicitly,  But java uses pointer implicitly: Java use pointers for manipulations of references but these pointers are not available for outside use. Any operations implicitly done by the language are actually NOT visible.
Pointers can do arithmetic, References can’t: Memory access via pointer arithmetic is fundamentally unsafe and for safe guarding, Java has a robust security model and disallows pointer arithmetic for this reason. Users cannot manipulate pointers no matter what may ever is the case.
- Pointing objects: In C, we can add or subtract address of a pointer to point to things. In Java, a reference points to one thing only. You can make a variable hold a different reference, but such c manipulations to pointers are not possible.
- References are strongly typed:  Type of a reference is much more strictly controlled in Java than the type of a pointer is in C. In C you can have an int* and cast it to a char* and just re-interpret the memory at that location. That re-interpretation doesn’t work in Java: you can only interpret the object at the other end of the reference as something that it already is (i.e. you can cast a Object reference to String reference only if the object pointed to is actually a String).
- Manipulation of pointers can be dangerous:  On one hand, it can be good and flexible to have control over pointers by user but it may also prove to be dangerous. They may turn out to be big source of problems, because if used incorrectly they can easily break assumptions that your code is built around. And it’s pretty easy to use them incorrectly.

You can make a scanner to scan the input of users in the console as follows : `Scanner input = new Scanner(Systen.in)`. Then you can check the inputs from the scanner rather easily as follows : `input.hasNext("[aeiou]")`. This means that our scanner scans and checks when a vowel appears first. Then you can access the next element as follows : `input.next`. 

You can compare dates as follows :  `dateStart.compareTo(dateEnd)`. Instead of setting `this.start = dateStart`, you can write instead the following : `this.start = new Date(dateStart.getTime())`. Similarly when you return data you make a new copy of what you want to copy. Instead of writing `return start`, you can write `return (Date) start.clone()`.

You can check if the float is not a number as follows :`Flot.isNan(argument)`, this returns a boolean. You can also use the for all way of iterating as follows :

`forall {int i in (0 : upperBound); a[i] != key }`

We can also have assertions, we have : `assert CONDITION : "string to be sent if condition not verified"`.

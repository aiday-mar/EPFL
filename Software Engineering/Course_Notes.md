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

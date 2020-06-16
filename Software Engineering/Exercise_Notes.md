# Exercise Notes

**Bootcamp**

Fill later

**Week 2**

In the basic exercise of the modularity example, we want to modularize the code and create separate methods to compute various parts rather than one big method. First of all we have a main method defined as follows `  public static void main(String[] args) throws FileNotFoundException { compute(); }`. In this main method we are calling another method called compute, which will do other calculations. Then in this computer method we actually return a listof Doubles. Here all operations are in separate methods.

We have in the LoadFromFile method that we instantiate a new file using the given filepath. Then we need to create a scanner that depends on the file at hand. This scanner checks whether the file has a double and if so returns this double. We use the following methods `scanner.hadNextDouble()` and `scanner.nextDouble()`. When you stop using the scanner you need to close the scanner `scanner.close()`. You can iterate over a list of doubles as follows : `for (double d: list) {}`. You can convert a double n to a string by writing `Double.toString(n)`.

But that's not what java.lang.Cloneable does. It doesn't even have any methods! Instead, it's a special interface you must implement if you want to be able to call super.clone() in your class without receiving an exception. As to what concerns the TCP : The issue is in the checksum: not only does it check the header and the payload, but it also checks a "pseudo-header" consisting among other things of the source and destination IPs.

Unfortunately, the only exact real number class in Java is BigDecimal, which is designed to have infinite precision and thus sacrifices poor performance. Some languages have a built-in type with specific precision that does not lose information in a given range, e.g. C#'s decimal.

Before Java 8, interfaces had only public abstract methods, while abstract classes could have both abstract and non-abstract methods. Classes could implement any number of interfaces and extend exactly one class.

Since Java 8, interfaces can have default methods, which are virtual: implementing classes can choose to override them or not, but the default implementation should make sense. This allows library developers to add features to their interfaces without breaking code that implements them. Before Java 8, adding a method to an interface was a breaking change: code that implemented the interface would not compile any more since it now lacked a method implementation.

This means that the line between interfaces and abstract classes is now rather blurry, with the main differences being that interfaces cannot have state and that a class can implement multiple interfaces but only extend a single class.

ArrayList, HashSet, and ArrayDeque are all Iterables, but that's it. Only one of them is a set (unordered, no duplicates), none of them are queues (first-in-first-out), an only one is a list (can have duplicate elements and supports insertion/removal/query at any index).

ArrayList, Vector, and LinkedList are both Iterables and Lists. They are not Sets since they have an order and allow duplicate elements, and certainly not Queues.

ConcurrentLinkedQueue, AbstractSet, and Properties are all Iterables, but that's it.

BufferedReader, OutputStreamWriter and GZIPInputStream can all be conceptually "closed", thus they should implement Closeable. But not all of them can receive input or output, thus they are not all InputStreams, Readables or Flushables (flushing is only doable on something you can write to).

PrintStream, OutputStreamWriter and FilterWriter all represent something you can write to, which mean they should be Flushables and Closeables but not InputStreams or Readables.

Integer, Path and ZonedDateTime are all things you can compare to other things of their category. They can also all logically be serialized. Thus they should implement Comparable and Serializable , but not Iterable (how would one iterate an integer or a date?) or Watchable (how would one watch an integer?).

BigDecimal, ThaiBuddhistCalendar and java.util.regex.Pattern do not have much in common, beyond the fact that they can all be represented in some way in a binary form, i.e. Serializable.

The Nobel Prize, the Turing Award and the Fields Medal are all awards. But they have different names, different criteria, and different regularity (the Fields Medal is only given once every four years). Thus, an interface representing them should expose these three concepts. Such an interface could also include a list of past winners, or a detailed description of the award.

The AuthenticationClient interface should contain:

A method to log an user in, given that user's credentials
A method to register a new user, given that user's information
A method to reset an user's password, given that user's e-mail

**Week 3**

The Optional container object helps provide optional values instead of null values. The search logic remains same except that we no longer return results but rather results wrapped inside optionals. This implies returning an optional of a result when there are available results. Otherwise, we return an empty Optional when there is no more result.

```
public static Optional<Result> search(String[] keywords) {
  Iterator<String> iterator = Internet.find(keywords);
  
  # meaning that there is at least one result
  if (iterator.hasNext()) {
    # the result must depend on the given iterator
    return Optional.of(new Result(iterator));
  }
  
  # otherwise if there is no result you return an empty optional
  return Optional.empty();
}
```
You can check the optionalvariable is present as follows :

```
Optional<Result> optional = Google.search(keywords);
while (optional.isPresent()) {
  Result result = optional.get();
  optional = result.next();
}
```

*Bad Input Parameter Checks and Exceptions*

You can also check the input parameters are not null by throwing otherwise a new IllegalArgumentException("the string to display in case of an exception"). You can also create a SengmentationFaultClass which extends the Exception class, and which has a constructor in which we call the constructor of the parent class. The implementation is :

```
public class SegmentationFault extends Exception {
  public SegmentationFault() {
    super();
  }
}
```


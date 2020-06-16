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

*Neutral return values*

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
When we try to access an index out of the bounds of the cstring we can define the following function which returns a randome ASCII character. 

```
public char get(int index) {
  if (index < 0 || index >= chars.length) {
    # you look at the length of the ascii vector, and take a random index in this length. This index is furthermore an integer.
    # then you find the value at that index and you return the whole element.
    return ascii.charAt(random.nextInt(ascii.length()));
  }
  # otherwise actually return the value when the index is not out of bounds. 
  return chars[index];
}
```

In a similar way we can change the set method as follows :

```
public void set(int index, char value) throws SegmentationFault {
  if (index < 0 || index >= chars.length) {
    throw new SegmentationFault();
  }
  chars[index] = value;
}
```

Also to get a string of the ascii characters, you need to create a new random element, and you initialize a new empty string called ascii, as well as an empty array of characters chars. This ascii essentially is juste the casting of the integers between 0 and 127 into characters : 

```
 private static Random random = new Random();
  private static String ascii = "";
  private char[] chars;

  static {
    for (int i = 0; i < 128; ++i) {
      ascii += (char) i;
    }
  }
 ```
Imagine you want to implement the string library. Then in the string copy method to copy one string into another. In our example we have that the \0 implies the end of the string. So while we have the element on that index is not \0 then in the destination, at the given index, you copy the given element in the source at that index. Then you augment the index by one integer. Otherwise you add an \0 in the last position. When the class returns an exception then you can write the definition of the class as being a public static class and then it returns a cstring and then the name of the method is strcat. This method returns a SegmentationFault.

In the strcat method we augment the i index until we reach the index at which we have the end of the string in the destination. Then you add into the next index of the destination string, the character at the next index of the source. In the strlen method you just increment the length while the element in the current index is not the end of string symbol \0.

In the strcmp method you are basically comparing two strings. For this you compare the characters at the respective indices of the two strings. If they are equal the characters then you can increase the index by one. And then if the two strings are not exactly equal then you return the difference between the two characters at the point where there is a divergence. Otherwise the return value is zero which means that the strings are equal.

*Callback-based error-processing routines*


A callback is a piece of executable code that is passed as an argument to other code, which can then execute the code passed in arguments by "calling it back" after completion. The invocation may be immediate (synchronous callback) or might happen at a later time (asynchronous callback).

In this exercise we restrict ourselves to using callbacks for error handling, but the concept can be applied to many other scenarios (such as asynchronous result delivery). An example of asynchronous callbacks is network-handling code in Androi apps: user actions which require communication over the network are often done asynchronously in the background instead of on the UI thread to avoid locking up the application while the user waits for a result from the network.

Conceptually, the interface to a callback generally looks like a function `callback(err, res)`:

1. The first argument of the callback is reserved for an error object. If an error occurred, it will be passed to the callback as the first `err` argument.
1. The second argument of the callback is reserved for any successful response data. If no error occurred, `err` will be set to null, and any successful data will be returned in the `res` argument.

In an object-oriented paradigm, a callback would be represented as an abstract class that provides two methods: one for any successful response data, and one for an error object.

Here is an example of a generic `Callback` interface in Java:

```java
interface Callback<T> {
  void onSuccess(T value);
  void onError(Exception e);
}
```

In this exercise, you are given a program that makes use of exceptions as its defense mechanism and your task will be to replace all exceptions with callbacks.

Callbacks are used for _delivering an error asynchronously_. The programmer passes a function or object—the callback—to a procedure. The latter invokes the callback sometime later when the asynchronous operation completes. Exceptions on the other hand are used for _delivering an error synchronously_. The programmer catches an error when some code throws an exception.

An example of the callback use is :

````
public void random(Callback<Joke> callback) {
  try {
    # here we have the onSuccess method which is called from the callback, and here you invoke a random joke in the parameter
    callback.onSuccess(service.random());
  } catch (NoJokeException e) {
    # in this case we call the onError method from the callback
    callback.onError(e);
  }
}
````

We can use the callback as follows :

```
# in here you need to specify a specific instance of the callback with some overriden methods onSuccess and onE
repository.random(new Callback<Joke>() {

  @Override
  public void onSuccess(Joke joke) {
    System.out.println(joke);
  }

  @Override
  public void onError(Exception e) {
    System.out.println(e);
  }
});
```

You can create a file reader from reading a file as follows : `BufferedReader reader = new BufferedReader(new FileReader(filename));`. You can check the input from this buffer as follows :

```
while ((line = reader.readLine()) != null) { ... }
```

In fact we have the following code :

```
package ch.epfl.sweng.defensive.error.processing.routine.store;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ch.epfl.sweng.defensive.error.processing.routine.model.Joke;

public class JokeStore {

  private static JokeStore store;
  private List<Joke> jokes = new ArrayList<>();
  private Random random = new Random();

  private JokeStore() {
    String path = JokeStore.class.getProtectionDomain().getCodeSource().getLocation().getPath();
    // the s after the % symbol must mean that this is a string
    String filename  = String.format("%s%s", path, "jokes.txt");
    try {
      BufferedReader reader = new BufferedReader(new FileReader(filename));
      String line, statement = "";
      // supposing that the nextline in the file is not null
      while ((line = reader.readLine()) != null) {
        if (!line.isEmpty()) {
          statement += " " + line;
        } else {
          // suppose that the line is empty, then if the whole statement is not empty
          if (!statement.isEmpty()) {
            jokes.add(new Joke(statement)); 
            // then in this case you add a new joke which is based on this statement
            statement = "";
          }
        }
      }
      reader.close();
    } catch (Exception e) {
      System.out.println(e);
    }
  }

  public static JokeStore get() {
    if (store == null) {
      // when the store is null then you need to create a new joke store
      store = new JokeStore();
    }
    return store;
  }

  public Joke random() {
    // here you are taking the size of the joke, then you choose a random index in this size and get
    // the corresponding element.
    return jokes.get(random.nextInt(jokes.size()));
  }
}
```

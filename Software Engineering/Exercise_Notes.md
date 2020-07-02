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

We can define a subclass of the exception as follows :

```
public class NoJokeException extends Exception {
  // we have here the constructor of the class
  public NoJokeException(String message) {
    // then we call the constructor of the parent class and we can input into it a string
    super(String.format("no kidding: %s", message));
  }
}
```

*How defensive programming affects code coverage*

This exercise help you understand the impact that defensive programming has on code coverage. Code coverage is a measure used to describe the degree to which the source code of a program is executed when a particular test suite runs.

On the other hand, defensive programming protects your code from invalid inputs and barricades your program to contain the damage caused by errors. We have the following keywords that can be used in conjunction with the Optional object. 

```
Optional<String> opt = Optional.empty();
Optional<String> opt = Optional.of("Hello, world!");
Optional<String> opt = Optional.ofNullable(object);
if (opt.isPresent()) { /* ... */ }
String str = opt.get();
```
An if statement actually adds a node in the decision tree of your program. In other words, it introduces two branches: one branch for when the condition is true, and one branch for when the condition is false. You gain robustness at the cost of complexity. When you try to run the code in a functional way, the coverage is much higher. 

```
public class CaseStudy {
  public static void demonstrate() {
    String code = "A code";
    Courses.findByCode(code)
      .flatMap(Course::getLecturer)
      .flatMap(Lecturer::getName)
      .ifPresent(System.out::println);
  }
}
```
Why do you have to use the flatMap method above ? I guess that here you are calling the respective functions each time on the return value of the previous code. By using Optional, and never working with null, you can avoid null checks altogether. Furthermore, with the functional paradigm, you can also avoid adding new branches in the decision tree of your program.

*CSV sanitization*

Here args is the variable which denotes the user input. The last line indicates that the program is terminated. We have 

```
if (args.length < 4) {
  System.out.println("usage : -i <input> -o <output> -v");
  System.exit(1);
}
```
Then we have the following code which is used to check if we did input '-i' into the console.

```
if (!args[0].equals("-i")) {
  System.out.println("missing input file");
  System.exit(1);
}
String inputFileName = args[1];
```
Similarly we have :

```
if (!args[2].equals("-o")) {
  System.out.println("missing output file");
  System.exit(1);
}
String outputFileName = args[3];
```
We have the following stream definition :

```
try (Stream<String> stream = Files.lines(Paths.get(inputFileName))) {
  BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputFileName));
  stream.forEach(line -> {
    //...
```

In the above we have a stream. This stream works with Strings. The stream is equal to a file instance from which we call the lines method and inside we pass the full path. Then we create a buffered writer from the output path. We define the following Format class. We have :

```
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Format {
  // the constructor is empty
  private Format() {}
  // the method returns a boolean, and takes in two arguments. There is a pattern which is made from compiling the regex, there is
  // also an instance of the matcher class which checks whether the value matches the given regex pattern. The next line returns a
  // boolean.
  public static boolean matches(String value, String regex) {
    Pattern pattern = Pattern.compile(regex);
    Matcher matcher = pattern.matcher(value);
    return matcher.matches();
  }
}
```
The full main file can be written as so :

```
package ch.epfl.sweng.defensive.garbage.in.non.garbage.out;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.logging.Logger;
import java.util.stream.Stream;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {
  
  // the logger allows you to log some results into the log. For some reason it is declared to be a static variable. What is the
  // argument which is used.
  static Logger logger = Logger.getLogger("");
  
  // once again the variable is static, you do not need to instantiate this variable. We have here an array of strings.
  static final String[] columns = {"datetime", "ip address", "user-agent", "url"};
  // the regex patterns are also given by strings in an array
  static final String[] regexes = {
    "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$",
    "^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
    "^.+$",
    "^(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?$"
  };
  
  public static void main(String[] args) {
    // here the args denotes the arguments which are the input from the user in the console.
    if (args.length < 4) {
      System.out.println("usage : -i <input> -o <output> -v");
      // we quit the program by writing the code below
      System.exit(1);
    }
    
    if (!args[0].equals("-i")) {
      System.out.println("missing input file");
      System.exit(1);
    }
    // in which case since we know there are already 4/5 arguments then the second one must be the file name 
    String inputFileName = args[1];
  
    if (!args[2].equals("-o")) {
      System.out.println("missing output file");
      System.exit(1);
    }
    String outputFileName = args[3];
    
    // the case when we have a verbose flag put at the end, and only when there are exactly five arguments.
    final Boolean verbose = args.length == 5 && args[4].equals("-v");

    try (Stream<String> stream = Files.lines(Paths.get(inputFileName))) {
      // there is a method in the files class which allows us to create a new buffered writer.
      BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputFileName));
      // then we iterate over the lines of the stream
      // use a lambda function inside of the parentheses of the forEach method.
      stream.forEach(line -> {
        if (!line.isEmpty()) {
          // when the line is empty, you create a new array of strings where you take the line and you split it according to the
          // comma
          String[] values = line.split(",");
          // meaning that we have exactly one value per column
          if (values.length != columns.length) {
            if (verbose) {
              // this info method likely allows us to log information into the computer
              logger.info(String.format("ignored {%s} : missing values", line));
            }
          } else {
            Boolean valid = true;
            for (int i = 0; valid && i < values.length; ++i) {
              // here we can check whether the ith value of the array matches the ith regex array
              if (!Format.matches(values[i], regexes[i])) {
                valid = false;
                if (verbose) {
                  // now you log the fact that there is a string which does not match the requires regex
                  logger.info(String.format("ignored {%s} : invalid {%s}", line, columns[i]));
                }
              }
            }
            // meaning that we still have that all the values match the required regex patterns 
            if (valid) {
              try {
                // in this case you write the line to the buffered writer
                writer.write(line);
                writer.newLine();
              } catch (IOException e) {
                // you print the error of the stack trace
                e.printStackTrace();
              }
            }
          }
        }
      });
      // here you need to close the buffered writer, as well as the stream.
      writer.close();
      stream.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
```

*Loop Invariants*

We must show three things about a loop invariant I:

- Initialization: I is true prior to the first iteration of the loop.
- Maintenance: If I it is true before an iteration of the loop, I remains true throughout the body of the loop until just before the next iteration.
- Termination: When the loop terminates (exits), I gives us a useful property that helps to show that the algorithm is correct.

The insertion sort can be implemented as follows :

```
public class InsertionSort {
  // We have here a static class which depends on the type C. It is also a void function, which takes as a parameter an array and 
  // a comparator 
  public static <C> void sort(C[] array, Comparator<C> comparator) {
    int j = 1;
    while (j < array.length) {
      // meaning that the jth element of the array is of type C
      C key = array[j];
      int i = j - 1;
      // when you compare the ith element of the array and the key and find that the value is positive
      while (i > -1 && comparator.compare(array[i], key) > 0) {
        array[i + 1] = array[i];
        i = i - 1;
      }
      array[i + 1] = key;
      ++j;
    }
  }
}
```

To verify the loop invariant first you want to define a helper method which will be called hold. It is given by :

```
public class LoopInvariant {
  // the third parameter is also the iteration number
  public static <C> boolean hold(C[] array, Comparator<C> comparator, int iteration) {
    assert iteration <= array.length : "iteration > array.length";
    for (int i = 0; i < iteration - 1; ++i) {
      if (comparator.compare(array[i], array[i + 1]) > 0) {
        return false;
      }
    }
    return true;
  }
}
```

*Interface Contracts*

Such a list contains objects that can be treated as Students. `List<? super Student>` expresses an upper bound on the type of the objects in the list. In such a list you can find Objects or Students — any of the superclasses of `Student` including itself, but never a subclass (e.g. no `SwEngStudent`). Here SwEngStudent is not accepted because it is not a super class of Student.

Consider now the two lists :

```
List<? extends Student> l1 = ...;
List<? super Student> l2 = ...;
```

Now you can perform the following `Student s1 = l1.get(0);` operation because we know that the list contains students or subclasses of students. However for the same reason the following code is not valid : `Student s2 = l2.get(0)`. Conversely the following operation `l1.add(new Student())` is not safe because you may want to extract specifically SwEngStudents and then you find a general Student. On the contrary if you consider the following operation `l2.add(new Student())`, this is safe, because this list can contain any of the superclasses of the students and the list will not be expected to contain anything more specific than a student.

*Regex Dos*

First we do an overview of the regex syntax. Most patterns use normal ASCII, which includes letters, digits, punctuation and other symbols on your keyboard like %#$@!, but unicode characters can also be used to match any type of international text. Any digit from 0 to 9 can be written with \d. Similarly, there is the concept of a wildcard, which is represented by the . (dot) metacharacter, and can match any single character (letter, digit, whitespace, everything). You may notice that this actually overrides the matching of the period character, so in order to specifically match a period, you need to escape the dot by using a slash \. accordingly.

There is a method for matching specific characters using regular expressions, by defining them inside square brackets. For example, the pattern [abc] will only match a single a, b, or c letter and nothing else. To represent this, we use a similar expression that excludes specific characters using the square brackets and the ^ (hat). For example, the pattern [^abc] will match any single character except for the letters a, b, or c. Luckily, when using the square bracket notation, there is a shorthand for matching a character in list of sequential characters by using the dash to indicate a character range. For example, the pattern [0-6] will only match any single digit character from zero to six, and nothing else. And likewise, [^n-p] will only match any single character except for letters n to p. Multiple character ranges can also be used in the same set of brackets, along with individual characters. An example of this is the alphanumeric \w metacharacter which is equivalent to the character range [A-Za-z0-9_] and often used to match characters in English text.

We've so far learned how to specify the range of characters we want to match, but how about the number of repetitions of characters that we want to match? One way that we can do this is to explicitly spell out exactly how many characters we want, eg. \d\d\d which would match exactly three digits. A more convenient way is to specify how many repetitions of each character we want using the curly braces notation. For example, a{3} will match the a character exactly three times. Certain regular expression engines will even allow you to specify a range for this repetition such that a{1,3} will match the a character no more than 3 times, but no less than once for example. This quantifier can be used with any character, or special metacharacters, for example w{3} (three w's), [wxy]{5} (five characters, each of which can be a w, x, or y) and .{2,6} (between two and six of any character).

For example, to match the donations above, we can use the pattern \d* to match any number of digits, but a tighter regular expression would be \d+ which ensures that the input string has at least one digit.
These quantifiers can be used with any character or special metacharacters, for example a+ (one or more a's), [abc]+ (one or more of any a, b, or c character) and .* (zero or more of any character). It would seem like also you are matching them in a sequence.

Another quantifier that is really common when matching and extracting text is the ? (question mark) metacharacter which denotes optionality. This metacharacter allows you to match either zero or one of the preceding character or group. For example, the pattern ab?c will match either the strings "abc" or "ac" because the b is considered optional. Similar to the dot metacharacter, the question mark is a special character and you will have to escape it using a slash \? to match a plain question mark character in a string.

We can use the meta-character '\d' to match the number of files and use the expression \d+ files? found\? to match all the lines where files were found. Here when you want to have a space then you actually write the space. The most common forms of whitespace you will use with regular expressions are the space (␣), the tab (\t), the new line (\n) and the carriage return (\r) (useful in Windows environments), and these special characters match each of their respective whitespaces. In addition, a whitespace special character \s will match any of the specific whitespaces above and is extremely useful when dealing with raw input text.

We have to match only the lines that have a space between the list number and 'abc'. We can do that by using the expression \d\.\s+abc to match the number, the actual period (which must be escaped), one or more whitespace characters then the text.

One way to tighten our patterns is to define a pattern that describes both the start and the end of the line using the special ^ (hat) and $ (dollar sign) metacharacters. In the example above, we can use the pattern ^success to match only a line that begins with the word "success", but not the line "Error: unsuccessful operation". And if you combine both the hat and the dollar sign, you create a pattern that matches the whole line completely at the beginning and end. Note that this is different than the hat used inside a set of bracket [^...] for excluding characters, which can be confusing when reading regular expressions.

The expression 'Mission: successful' will match anywhere in the text, so we need to use the starting and ending anchors in an expression ^Mission: successful$ to only match the full string that starts with 'Mission' and ends with 'successful'.

Regular expressions allow us to not just match text but also to extract information for further processing. This is done by defining groups of characters and capturing them using the special parentheses ( and ) metacharacters. Any subpattern inside a pair of parentheses will be captured as a group. In practice, this can be used to extract information like phone numbers or emails from all sorts of data. Imagine for example that you had a command line tool to list all the image files you have in the cloud. You could then use a pattern such as ^(IMG\d+\.png)$ to capture and extract the full filename, but if you only wanted to capture the filename without the extension, you could use the pattern ^(IMG\d+)\.png$ which only captures the part before the period.

We only want to capture lines that start with "file" and have the file extension ".pdf" so we can write a simple pattern that captures everything from the start of "file" to the extension, like this ^(file.+)\.pdf$.

This expression requires capturing two parts of the data, both the year and the whole date. This requires using nested capture groups, as in the expression (\w+ (\d+)). We can alternatively use \s+ in lieu of the space, to catch any number of whitespace between the month and the year. This will return for example `Jan 1987` and `1987`. You can also decide to capture different parts of a phrase by putting the parentheses separately. Suppose you apply the following regex '(\d+)X(\d+)' on the input 1280x720, then the output is 1280 and 720.

Specifically when using groups, you can use the | (logical OR, aka. the pipe) to denote different possible sets of characters. In the above example, I can write the pattern "Buy more (milk|bread|juice)" to match only the strings Buy more milk, Buy more bread, or Buy more juice. An example is : 'I love (cats|dogs)' which matches both of the variants.

We have already learned the most common metacharacters to capture digits using \d, whitespace using \s, and alphanumeric letters and digits using \w, but regular expressions also provides a way of specifying the opposite sets of each of these metacharacters by using their upper case letters. For example, \D represents any non-digit character, \S any non-whitespace character, and \W any non-alphanumeric character (such as punctuation). Depending on how you compose your regular expression, it may be easier to use one or the other. Additionally, there is a special metacharacter \b which matches the boundary between a word and a non-word character. It's most useful in capturing entire words (for example by using the pattern \w+\b).

**Week 4**

*Find the bugs*

Now in the Person class we are checking that age < 17 rather than age < 18. The test below then should fail :

```
@Test
// notice that below the name of the test is put directly into the name of the method
void seventeenYearOldPeopleAreMinors() {
    assertThat(new Person("Carmen", "Sandiego", 17).isMinor(), is(true));
}
```

In the addressbook example we have that the last name of the person is not taken into account :

```
@Test
void differentLastNamesButSameFirstNamesAndAgesAreNotConfused() {
    AddressBook book = new AddressBook();
    book.setAddress(new Person("Alan", "Turing", 99), "Secret Enigma-cracking lab");
    book.setAddress(new Person("Alan", "Rickman", 99), "Potions-brewing room");
   
    // sometimes you may want to verify two conditions which you can do with both().and()
    // inside of each brackets a true value must be returned. For this we probably must have defined a custom
    // containsString() method which returns a boolean true or false. 
    assertThat(book.toString(),
           both(containsString("Alan Turing: Secret Enigma-cracking lab"))
          .and(containsString("Alan Rickman: Potions-brewing room"))
    );
}
```

The WorkLog class uses ZonedDateTime.now(), which depends on the real system clock. This is a problem for tests because they could pass or fail depending on the date or time! Instead, the WorkLog class needs to take a Clock class in its constructor, which should be stored as a private field and then passed to the now method, which has an overload accepting a Clock. To this end, we can create a FakeClock which extends the Clock as follows : 

```
private final class FakeClock extends Clock {
    private Instant instant;

    void setInstant(Instant instant) {
        this.instant = instant;
    }

    @Override
    public ZoneId getZone() {
        return ZoneId.of("UTC");
    }

    @Override
    public Clock withZone(ZoneId zoneId) {
        // this method therefore should not be used 
        throw new RuntimeException("This method should not be called");
    }

    @Override
    public Instant instant() {
        return instant;
    }
}
```

Now if the start and the end are not on the same day, there can be an error in terms of the number of hours worked. Instead we can decide to replace the current code with the following code :

```
int hours = (int) start.until(stop, ChronoUnit.HOURS);
```

In the above we have case the time spent working as an integer. We have the start and then we have the until method wich accets a stopping time and display the time between in terms of hours. 

*Adapt the code*

You will change the code to use dependency injection instead. The idea is simple: instead of instantiating the objects and services that a class depends on (i.e., its "dependencies"), a class should receive them as parameters to its constructor (i.e., get them "injected"). Those dependencies should be interfaces, not specific implementations. For example instead of writing `new RealHttpClient()` to instantiate the interface, you can write : 

```
private final HttpClient client;

public WeatherService(HttpClient client) {
    this.client = client;
}
```

Not only does GoogleService not implement an interface, but its signIn method is static! To avoid this issue, create the interface yourself, then use the Adapter pattern to create the "real" implementation. The Adapter pattern is a simple concept: create adapter code so that a type can be used as if it implemented an interface. This is just like real-life adapters, where a Swiss laptop charger can be used in Japan with an adapter that "implements" the Swiss electrical "interface" on top of the Japanese one. For instance, in the Google sign-in example, the code could look like this:

```
class MyUser {
  // ... properties of GoogleUser that you want to expose to the app ...
  // This class exists so that the rest of the app doesn't depend on GoogleUser;
  // after all, Google is just one way to sign in, you could in the future add Facebook or Twitter.
}

interface SignInService {
  MyUser signIn();
}

// below is the adapter which is implementing the sign in service 
class GoogleSignInAdapter implements SignInService {
  public MyUser signIn() {
    GoogleUser user = GoogleService.signIn();
    // ... convert 'user' to an object of class MyUser, and return it ...
  }
}
```

You can create an interface for getting a user's position as follows :

```
public interface LocationService {
    Position getUserPosition();
}
```

We implement the interface by writing a specific service which will extend this interface as follows :

```
public final class GeolocatorLocationService implements LocationService {
    
    // the geolocator is initialized here, it's a final variable because it can not be changed
    private final Geolocator locator = new Geolocator();

    @Override
    public Position getUserPosition() {
        return locator.getUserPosition();
    }
}
```

Now while before we had the following code for TreasureFinder : 

```
public class TreasureFinder {

    private final Geolocator geolocator;



    // There MUST be a parameterless constructor,

    // it is used by our Super-Fancy-Framework-That-Does-Not-Support-Parameters™

    public TreasureFinder() {

        geolocator = new Geolocator();

    }



    public String getHint(Position treasurePos) {

        Position userPos = geolocator.getUserPosition();

        if (userPos.latitude > 70) {

            return "Nope, the treasure is not at the North Pole.";

        }



        // Not accurate because of the Earth's curvature. Better calculation coming next sprint!

        double diff = Math.sqrt(Math.pow(treasurePos.latitude - userPos.latitude, 2) + Math.pow(treasurePos.longitude - userPos.longitude, 2));



        if (diff < 0.005) {

            return "You're right there!";

        }



        if (diff < 0.05) {

            return "Close...";

        }



        if (diff < 0.5) {

            return "Not too far.";

        }



        return "Far away.";

    }

}
```

We can rewrite this TreasureFinder using the location service to get : 

```
private final LocationService locationService;

public TreasureFinder() {
    // new instance of the service 
    locationService = new GeolocatorLocationService();
}

public TreasureFinder(LocationService locationService) {
    // here we have a dependency injection 
    this.locationService = locationService;
}
```

Now we may want to test the above code as follows :

```
@Test
public void northPoleTriggersEasterEgg() {
    // here we have a location specified in the parentheses in the service returning
    LocationService locator = serviceReturning(80.0, 1.0);
    // the treasure finder depends on the locator which takes in 2D coordinates 
    TreasureFinder finder = new TreasureFinder(locator);
    // in the assert method we have the new position and we get the hint associated which returns us a string 
    // which is related to this coordinate position. Then we compare this string against the other string in the 
    // is() method
    assertThat(finder.getHint(new Position(1.0, 1.0)), is("Nope, the treasure is not at the North Pole."));
}

// here we input a position given by a lattitude and a longitude in the parentheses of the serviceReturning method
private static LocationService serviceReturning(double lat, double lon) {
    return new LocationService() {
        // you return an instance of this location service class in which you want to override the method below
        // this method return a position with the right parameters
        @Override
        public Position getUserPosition() {
            return new Position(lat, lon);
        }
    };
}
```

*Dependency Injection on Adroid*

You can use a static factory for dependencies as follows :

```
public final class HttpClientFactory {
    private static HttpClient client = new RealHttpClient();

    public static HttpClient getClient() {
        return client;
    }
    
    // why do we just not access this.client and we write the full name of the class ? 
    public static void setClient(HttpClient client) {
        HttpClientFactory.client = client;
    }
}
```

We use the factory as follows :

```
public WeatherService() {
    this.client = HttpClientFactory.getClient();
}
```

We can write tests now :

```
@Test
void questionMarksMeanItsRainingMen() {
    HttpClientFactory.setClient(new HttpClient() {
        // you can override any methods in the HttpClient class
        // when the string throws an exception you can write this in the definition of the method
        @Override
        public String get(String url) throws IOException {
            // you return a string which consists in three question marks 
            return "???";
        }
    });

    WeatherService service = new WeatherService();
    
    // you use the service weather to find the current weather, and you use hamcrest to compare this to some parameter
    assertThat(service.getWeatherToday(), is(Weather.ITS_RAINING_MEN_HALLELUJAH));
}
```

Mocking libraries help you write fake dependencies, also called mocks, by making it easy to create implementations of any dependency interface that return the values you want, instead of having to write an implementation of the interface every time.

```
@Test
void questionMarksMeanItsRainingMen() throws IOException {
    HttpClient client = mock(HttpClient.class);
    when(client.get("http://example.org/weather/today")).thenReturn("???");
    WeatherService service = new WeatherService(client);

    assertThat(service.getWeatherToday(), is(Weather.ITS_RAINING_MEN_HALLELUJAH));
}
```

Fuzzing is an advanced testing technique consisting of sending random inputs to a program to find bugs. While this may seem odd at first glance, it has shown to be very effective especially in low-level languages where fuzzing finds crashes and undefined behavior quickly. Now that you have learned about fuzzing, download AFL, one of the most well-known fuzzers, and find bugs in the LAVA-M corpus of buggy programs. Or find bugs in your favorite program, and report them to the developers!

**Week 5**

When you have messed up the commit message you can write : `$ git commit --amend`. To view previous commits and messages you can run `$ got log`. To view a specific commit, copy it's hash ID and run : `$ git show <commit>`. Here we can find the version of java using : `javac -version`. You can check the jdb debugger are installed by writing : `jdb`. Where we have an array of integers : `int[] numbers = new int[]{ 1, 2, 3, 4, 5};`. You can find the length with numbers.length . You can run the program with javac, and run the program with java. 

First, recompile your program with the -g flag. This tells the compiler to add extra information to the class file so that the debugger is aware of local stack variables. You can add breakpoints in jdb using : `stop in <classname>`.

The cont command will continue execution, either until another breakpoint is hit, or the program terminates (or crashes). At any time you can use the help command to see all available commands in jdb.

Here we have to pay attention to the merge keyword. Merging is basically porting the code differences from one branch to another so that code features that are available in one branch are also available in the other one. There is one master branch, which can be used for release or keeping a stable version of the application and a develop branch, which is used for code development such as adding new features or bug fixes. When the developer is committing to a branch, that branch has to be checked out and the HEAD should point to the latest commit in that branch. We can run the following commands :

```
echo "# swengweek1" >> README.md  
git init  
// add this file 
git add README.md  
// the -m is the label for the message
git commit -m "first commit"  
// where here we have the url of the origin
git remote add origin https://github.com/<user-id>/<repo-id>.git 
// push the urls to the master 
git push -u origin master 
```
Commit and push the files :

```
git add *.java
git commit -m "adding java files"  
git push  
```

List all the branches we have : `git branch -a`. The previous command shows all the branches. To create a branch you can write : `git branch newbranch`. Now to delete the branch you have to write : `git branch -d newbranch`. To make sure you are working on the actual branch is : `git checkout fix-sort-bug`. Alternative method: Instead of first creating a branch, then checking out, you can do it with one command. This can be written with : `git checkout -b fix-sort-bug`. We can commit all the files to the QuickSort file as follows : `git log --follow -- QuickSort.java`. We can see all the commits for the current branch : `git log`. Now we want to merge the branch into the main branch :

```
git checkout master
git merge fix-sort-bug 
git branch -d fix-sort-bug  
git push
```
Now if we want to rollback completely to a previous state : `git reset --hard <commit-id>`. You can also view the differences between the versions as follows for example `git diff <commit-id1> <commit-id2>`.

0. Breakpoints: Code line locations to pause the execution of the program. The developer can execute expressions and view the state of the program.
1. Step-Over: Execute the method call and skip to the next statement without getting into inner details of the method.
2. Step-Into: Execute a statement, if it is a method call, go inside the method.
3. Step-Out: Finish the method call, return from it to the caller method and go to the next statement.
4. Resume: Resume until the next breakpoint, if no breakpoint then continue until the end of the program execution.
5. Re-run: Restart the debugged program.
6. View breakpoints: You can observe and modify all breakpoints.

*Quiz*

Large software projects miss their deadlines more often than small projects because the management task is more difficult and prone to wrong decisions as the project gets bigger. There should be no discussion during the daily scrum meeting. The Backlog should not change during the Sprint, it should be discussed during the Sprint Planning Meeting and Sprint Review. The burnout chart provides the information of an estimation of the amount of work left for the sprint and the average team daily performance. In every bar of the burnout chart you can find the parts of user stories that remain to be done. Now the product owner manages the product backlog. The product owner manages the product backlog. Who can move an item from the Product Backlog to the Sprint Backlog? The deveopment team. The role of the scrum master is to serve the Product Owner by helping them communicate with the Development Team, to serve the Development Team by coaching them in self-organization and flexibility, to serve the Development Team by removing impediments. The development team should speak in the daily scrum.

The Sprint Review has to show the Increment to stakeholders and the Product Owner so that they can communicate their expectations. The outcome is a revised Product Backlog that is ready for the Spring Planning meeting. The Backlog can be adjusted further by the Product Owner, if necessary. The Sprint Retrospective is for the Scrum Team to inspect what worked and what didn't during the Sprint. The goal of the Sprint Retrospective is to identify improvements that the Scrum Team can implement in the next Sprint. The Scrum Team can update its definition of Done during the Sprint Retrospective.

**Week 8**

There are no slides in between. There are various refactoring methods : push up common methods/properties of two subclasses, pull down a specific method/property to subclass, create a common class to all the methods/properties, divide the class into different subclasses. Code smell is when you have unreadable code, complex structures, or useless structures. Refactoring is made to simplify this code, make it more readable etc. The code can also become more maintainable, it reduces coupling. You can also delete classes that have no functionality. When your class uses mostly methods of other classes this is called feature envy. Refactoring can be dangerous, it can be expensive, and you can end up wasting time on it. A method's body should be just as clear as its name. You can return a boolean in a method as follows :

```
private in getRating() {
  return (moreThanFiveLateDeliveries()) ? 2 : 1;
}
```

Also instead of using a constant you can use a query, or a call to a method instead. You can create a list as follows too : `List<String> candidates = Arrays.asList(1,2,3)`. You can check whether this array contains something with : `candidates.contains(1)`.

**Week 9**

*Identifying code smell*

In the data clump you have properties in your class that are not necessarily related to the object you want to describe directly, so you can extract some properties and create two classes instead. Some methods are too long and so you can split the method into smaller methods. Similarly you can break up a specific computation into different parts by writing a simple expression per line.

The whole point of objects is that they are a technique to package data with the processes used on that data. A classic smell is a method that seems more interested in a class other than the one it is in. The most common focus of the envy is the data.

*Duplication*

Duplication of code can be in the same class/method, in the sibling class, in a different class or there could be two methods that could be considered similar. Suppose that a scanner is created in every single method of the same class then you can extract this scanner into a static field, and make it available to all the methods.

*Identifying refactoring techniques*

The algorithm used here didn't really make sense (why would you compute a minimum if the name of the method was getMostValuableMember ?). The method was pretty long so extracting a part of it into a separate method makes it more concise. It is also easier to spot a potential bug as one can test the sumTransactions method separately.

Explanation: The getCurrentlyUsedVersionNumber method was being called many times, especially inside the for-loop. Imagine the returned value requires performing expensive computations or needs to fetch the information from a distant server. Getting the value only once and storing it inside a variable reduces cost. The name of the mentioned method was clearly too long, and was renamed to something simpler that describes it's role just as well. The VERSIONS array stored the information about each version in arrays of ints. This hurts the understandability of the code "what was stored in [1] again ??". A Version object with explicit methods such as setStatus can be extracted. Finally, the magic numbers (i.e. 0x3) aren't meaningfull and can easily be misstyped. They have been replaced by constants in an enum. Therefoer the 4 refactoring methods used are : extract the variable, rename the method, extract the class and replace maginc numbers with symbolic constants.

*More refactoring*

Sometimes we can just use a simple string instead of using an object. On the contrary when three attributes specify one type of characteristic then you can extract these into a class for example. You can specify constants as follows : `private static final double FLOOR_HEIGHT = 10`. We have the following method from the Math class : `Math.toRadians(...), Math.cos(...), Math.sin(...), Math.atant2(...), Math.sqrt(...), Math.pow(... , ...)`. You can code the two methods isAvailable, and, isAvailableAt the following way :

```
public boolean isAvailable() {
    // where you need to input the urrent time into the method 
    return isAvailableAt(TimeSlot.now());
}
    
public boolean isAvailableAt(TimeSlot slot) {
    // here you use an instance of a class which we call occupancies
    // and we call the containsKey method from this instance, and using the passed in time slot.
    return occupancies.containsKey(slot);
}
```

Now we want to refactor the following method :

```
// where you map a timeslot to a specific course
private Map<TimeSlot, Course> occupancies;

public Course.TYPE mostCommonCourseType() {
        int nbMath, nbArt, nbEnglish, nbHistory, nbGeography;
        Course.TYPE currentBest = null;
        int currentMax = -1;
        
        // you consider all the second column that contains the courses and we iterate over these courses 
        for (Course c : occupancies.values()) {
            // where the getType returns an instance of the TYPE class
            switch(c.getType()) {
                case MATH:
                    nbMath++;
                    if(nbMath > currentMax) {
                        currentMax = nbMath;
                        currentBest = MATH;
                    }
                    break;
                case ART:
                    nbArt++;
                    if(nbArt > currentMax) {
                        currentMax = nbArt;
                        currentBest = ART;
                    }
                    break;
                case ENGLISH:
                    nbEnglish++;
                    if(nbEnglish > currentMax) {
                        currentMax = nbEnglish;
                        currentBest = ENGLISH;
                    }
                    break;
                case HISTORY:
                    nbHistory++;
                    if(nbHistory > currentMax) {
                        currentMax = nbHistory;
                        currentBest = HISTORY;
                    }
                    break;
                case GEOGRAPHY:
                    nbGeography++;
                    if(nbGeography > currentMax) {
                        currentMax = nbGeography;
                        currentBest = GEOGRAPHY;
                    }
                    break;
                default:
                    throw new Error("Undefined course type !");
            }
        }
        return currentBest;
    }
```

A more concise method is the following method :

```
public Course.TYPE mostCommonCourseType() {
    // here what we are likely doing is taking the Course.Type type and we are saying that the size of this array
    // will be the length of the second column of this map
    Integer[] counts = new Integer[Course.TYPE.values().length];
    // now that we know what the length should be we are going to fill in this array with zeroes
    Arrays.fill(counts, 0);
    // because we know that this map contains courses as the type of the variable in the second column
    for (Course c : occupancies.values())
        // The java.lang.Enum.ordinal() method returns the ordinal of this enumeration constant 
        // (its position in its enum declaration, where the initial constant is assigned an ordinal of zero).
        counts[c.getType().ordinal()]++;
    // you can stream the array, now to take the maximum likely you need to use a comparator to apply on the entries
    // therefore you call the compareTo method in the Integer class, and then you need to get() the result
    Integer max = Arrays.stream(counts).max(Integer::compareTo).get();
    int index = Arrays.asList(counts).indexOf(max);
    return Course.TYPE.values()[index];
}
```

**Week 10**

*Exercise 1*

Here, the subject is the DataStore and the observers are the different kind of Phone and Screen. When the subject changes, we want the observers to be notified in order to update their content to display the newly updated data. Consider the following example of an observable object class :

```
import java.util.ArrayList;

/**
 * This class represents an observable object. It can be subclassed to represent an object that the application wants to have observed.
 * A subject can have one or more observers. An observer can be any object that implements the Observer interface.
 */

public abstract class Subject {

    private ArrayList<Observer> observers;
  
    // here we have the constructor for an abstract class, since the class is abstract, then in front of the constructor we do not keep any keyword.
    Subject(){
        observers = new ArrayList<>();
    }

    void registerObserver(Observer o) {
        // adding into the array list 
        observers.add(o);
    }

    void removeObserver(Observer o) {
        // the index of this observer in this array list 
        int i = observers.indexOf(o);
        // meaning when this object exists in the array list 
        if (i >= 0) {
            observers.remove(i);
        }
    }

    void notifyObservers(String data) {
        for (int i = 0; i < observers.size(); i++) {
            // meaning we cast the ith element of the observers list as an Observer 
            Observer observer = (Observer) observers.get(i);
            observer.update(data);
        }
    }
}
```

Then you can have `public class DataStore extends Subject{}`. Then here we do not need to specify a constructor in the data store and we do not need to override the methods, we can however add some methods. You can extend classes and implement interfaces. Then in the class where you implement the interface you can override the methods from the interface. 

*Exercise 2*

We use the builder design pattern to make the code more manageable. Suppose you have a computer which has many properties you want to istantiate. Then you can create a long constructor with all the necessary properties to istantiate the class, however you may want to create a computer builder too. You can use the string builder as follows : 

```
StringBuilder sb = new StringBuilder(); 
sb.append(this.name); 
return sb.toString();
```

Then in the computer builder you have private variables to which we can assign default values. Then we have a build method which returns exceptions if one of the properties is null. Then you have an example of the set methods in the buider as follows : 

```
 public ComputerBuilder setDram(int dram) {
        this.dram = dram;
        // meaning we return this instance of the ComputerBuilder class
        return this;
    }
```

*Exercise 3*

We use the adapter design pattern to create a new class with the name NewFacebokPosterToFacebookPosterAdapter and instantiate it in the PosterTest class. The FacebookPoster class only has the method names without the definition of the body. Then you can have a FacebookPosterImpl class which implements the FacebookPoster interface from before.

*Exercise 4*

To remove elements from a table you can write : `table.remove(key)`. In the following class `public class DatabaseImplToKeyValueStoreAdapter implements KeyValueStore {...}`, we have a private DatabaseImpl, and in the constructor we use the method from the instance of the DatabaseImpl. Then you need to override the methods from the KeyValueStore, using the `@Override` keyword. There are a few methods you can use which are part of the Integer class `Integer.toString(key)` and `Integer.valueOf(result)`.

*Exercise 5*

In the abstract class Shape, you just need one method to be abstract, all the methods need not be abstract. You can also define an exception of an unsupported operation : `throw new UnsupportedOperationException("This operation is not supported!");`.

*Exercise 7*

In the absract class, the methods can have a body. You can have an abstract class extend an abstract class. Then this sub abstract class does not need to override the abstract methods of the original abstract class. However a simple class extending the abstract class must override the methods with no body in the original abstract class ? You can use the following to format strings : `String.format("An %s costs %.2f$", espressoWithSteamedMilk.getDescription(), espressoWithSteamedMilk.cost())`.

*Exercise 8*

When you have a subclass extending a class, then in the constructor you can call the constructor of the super class with `super(user.getUsername());`. You can have a `Jframe frame = new JFrame()` to show desktop applications, and a JButton("...") which takes as a parameter the string to show in the button. You can add listeners to the button where : `button.addActionListener(new AngelListener());` where we have that `class AngelListener implements ActionListener {...}`. Then you can add the button to the frame as follows : `frame.getContentPane().add(BorderLayout.CENTER, button);`. You can write :

```
frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
frame.getContentPane().add(BorderLayout.CENTER, button);
frame.setSize(300,300);
frame.setVisible(true);
```

And the action event listener is :

```
class AngelListener implements ActionListener {

  public void actionPerformed(ActionEvent event) {
    System.out.println("Don't do it, you might regret it!");
  }
}
```

**Week 11**

*Exercise 1*

We use a factory design pattern to be able to create different types of pizzas. The Static Factory Method is better suited in this case. Indeed, the goal here is to gain control over object (pizza) creation. This is achieved by delegating object creation from the constructor to static methods which act on behalf of the constructor. You have an abstract public pizza class. In a method prepare pizza you can call all the other methods in the same class for that specific instance :

```
public void preparePizza(){
    this.prepare();
    this.bake();
    this.cut();
    this.box();
}
```

Now you have the following subclass of the pizza class :

```
public class PepperoniPizza extends Pizza {
	public PepperoniPizza() {
		name = "Pepperoni Pizza";
		dough = "Crust";
		sauce = "Marinara sauce";
		toppings.add("Sliced Pepperoni");
		toppings.add("Sliced Onion");
		toppings.add("Grated parmesan cheese");
	}
}
```

Where the pizza class starts with :

```
abstract public class Pizza {
    protected String name;
    protected String dough;
    protected String sauce;
    protected ArrayList<String> toppings = new ArrayList<String>();
    ...
}
```

*Exercise 2*

We used to have 4 constructors with the same name and a different number of parameters. Now we create different static methods to create the corresponding loan, and in each method we have a corresponding call to the initial constructor. You also input where needed some default values into the initial constructor if in the corresponding call to the class you do not need these specific parameters defined. Then in this constructor you may create also after a corresponding instance of the capitalStrategy, or you create an instance of any subclass of the capitalStrategy. Meaning that here in the constructor you could create a CapitalStrategyTermLoan, a CapitalStrategyRevolver or a CapitalStrategyRCTL.

*Exercise 3*

We use the Abstract Factory design pattern to be able to create different types of computers. The example is :

```
public class PCFactory implements ComputerAbstractFactory {

    private String ram;
    private String hdd;
    private String cpu;

    public PCFactory(String ram, String hdd, String cpu){
        this.ram=ram;
        this.hdd=hdd;
        this.cpu=cpu;
    }

    @Override
    public Computer createComputer() {
        return new PC(ram,hdd,cpu);
    }
}
```

*Exercise 4*

We have the following code :

```
public interface House {
    Member createMember();
    Bastard createBastard();
}

public class HouseStark implements House {
    @Override
    public Member createMember() {
        return new StarkMember();
    }

    @Override
    public Bastard createBastard() {
        return new StarkBastard();
    }
}


public interface Member {
    void sayMotto();
}

public interface Bastard {
    void sayLastName();
}

public class StarkBastard implements Bastard {
    @Override
    public void sayLastName() {
        System.out.println("My last name is Snow.");
    }
}

public class StarkMember implements Member {
    @Override
    public void sayMotto() {
        System.out.println("Winter is coming.");
    }
}
```

*Exercise 5*

Now we want to log information into a logger. This logger is a singleton, there is only one instance of this class, so we make the logger a static class, then the constructor for the logger should be private as follows :

```
public class Logger {

    private static final Logger instance = new Logger();
    private Logger() {
    	// meaning we throw an exception because in this case there is already an instance of the logger that exists
        if (instance != null)
            throw new IllegalStateException("Already instantiated");
    }

    public static Logger getInstance() {
        return instance;
    }

    protected void print() {
        System.out.println("Logged");
    }
}
```

Otherwise the second way to create singleton classes is :

```
public enum Logger2 {

    INSTANCE;
    public Logger2 getInstance() {
        return INSTANCE;
    }
    
    protected void print() {
        System.out.println("Logged");
    }
}
```

And then to use the above two class you can write this in the App java class as follows :

```
public class App {
    public static void main(String[] args) {
        Logger logger = Logger.getInstance();
        logger.print();

        Logger2 logger2 = Logger2.INSTANCE;
        logger2.print();
    }
}
```

*Exercise 6*

Use the proxy design pattern to restrict access to all the files that contain the word "sensitive" in their filename. For this you can use the string.contains("...") method.

*Exercise 7*

Suppose we have a `private String output = ""`. Then you can add text to this string by writing for example `output += String.format("<b>%s</b>", part.getText())`.

*Exercise 8*

Once the process is finished, you will see several subfolders in the Project view on the left: most notably java where the logic lives, and res which will store our assets and layout declarations.

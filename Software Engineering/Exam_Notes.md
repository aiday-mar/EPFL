# Exam Notes

You can execute multiple tasks from a single build file. Gradle can handle that build file using gradle command. This command will compile each task in the order that they are listed and execute each task along with the dependencies using different options.

The Gradle Wrapper is an optional part of the Gradle build system. It consists of four files that you check into version control system. The *nix start script <your root project>/gradlew, the <your root project>/gradlew.bat Windows start script, <your root project>/gradle/wrapper/gradle-wrapper.jar which contains the class files for the wrapper and is started by the start scripts and <your root project>/gradle/wrapper/gradle-wrapper.properties which contains some configuration for the wrapper, for example which Gradle version to use to build the project.

In my opinion, each and every Gradle project, even the tiniest, should make use of the Gradle wrapper.

The Gradle wrapper makes sure your build is always run with the same Gradle version, no matter who executes the build and where or that Gradle is installed or not, as long as the one uses the Gradle wrapper to build the project. This means you can design your build for that Gradle version and be sure that the build will not fail, just because someone is using a different version of Gradle and thus is also an important step in build reproducibility.


**Exam 2019**

**Theory**

A proxy, in its most general form, is a class functioning as an interface to something else. The proxy could interface to anything: a network connection, a large object in memory, a file, or some other resource that is expensive or impossible to duplicate. In short, a proxy is a wrapper or agent object that is being called by the client to access the real serving object behind the scenes.

On android only views can handle user inputs, so MVC is not feasible.

Automated tesing and fuzzing only proves the presence of bugs.

When a new feature should be added you should ask the product owner to prioritize this in a sprint retrospective, he decides everything. 

Refactoring code can make the lines of code augment since we have increase abstraction, furthermore it should make code easier to understand the should make writing tests easier. 

The decorator pattern exposes the same interface as it wraps. A Decorator pattern can be used to attach additional responsibilities to an object either statically or dynamically. A Decorator provides an enhanced interface to the original object.

Unit tests are used on small pieces of code. End-to-end tests show how one program reacts to another one. Regression tests are written in response to bugs found previously.

FUZZ TESTING (fuzzing) is a software testing technique that inputs invalid or random data called FUZZ into the software system to discover coding errors and security loopholes. Data is inputted using automated or semi-automated testing techniques after which the system is monitored for various exceptions, such as crashing down of the system or failing built-in code, etc.

The Decorator pattern is about exposing the same interface but adding features; in this case, while it could be used to represent typo corrections as patches over incorrect menus, this is unlikely to be what users want sine they are changing text, not keeping track of all old versions. 

In software engineering, the singleton pattern is a software design pattern that restricts the instantiation of a class to one "single" instance. This is useful when exactly one object is needed to coordinate actions across the system. The term comes from the mathematical concept of a singleton.

The factory method design pattern handles these problems by defining a separate method for creating the objects, which subclasses can then override to specify the derived type of product that will be created. The factory method pattern relies on inheritance, as object creation is delegated to subclasses that implement the factory method to create objects.

**Practise : part 1**

We have the following link with the code : https://github.com/sweng-epfl/public/tree/master/exams/final/2019/src/main/java and https://github.com/sweng-epfl/public/tree/master/exams/final/2019/src/test/java

There are various types of exceptions such as the IllegalArgumentException as follows : `throw new IllegalArgumentException("some text");`. You can take a List of Strings called Command and cast this as a new ArrayList<>(commands). Suppose you want to create an unmodifiable list then you can write :

`Collections.unmodifiableList(new ArrayList<>(commands))`

Suppose you need to return a string representation of the view, then you can use a StringBuilder. You can also use the command `append` on it, to append some string to the builder. You can also append a line separator using the following command :

```
StringBuilder result = new StringBuilder();
result.append(System.lineSeparator());
```
You can also iterate over the strings in another string as follows : `for (String a : text) {}`. Then when you need to return the string built from the string builder you write : `return result.toString()`.

**Practise : part 2**

The basic structure of writing document comments is to embed them inside /** ... */. The Javadoc is written next to the items without any separating newline. Note that any import statements must precede the class declaration. 

To remember the results in the class we need to store it in some variable in the class. This is why we declare a variable `private final Map<String, List<Person>>`. Where here we have a string mapped to a list of persons.  Since the constructor has a Directory as a parameter that means that you need to initialize the directory and have it as a private final parameter in the class. You also need to check whether or not the passed paramaters, in this case the directory, are null or not. If this is the case then we throw an IllegalArgumentException. 

The TreeMap in Java is used to implement Map interface and NavigableMap along with the Abstract Class. The map is sorted according to the natural ordering of its keys, or by a Comparator provided at map creation time, depending on which constructor is used. This proves to be an efficient way of sorting and storing the key-value pairs. The storing order maintained by the treemap must be consistent with equals just like any other sorted map, irrespective of the explicit comparators. The treemap implementation is not synchronized in the sense that if a map is accessed by multiple threads, concurrently and at least one of the threads modifies the map structurally, it must be synchronized externally. Some important features of the treemap are:

This class is a member of Java Collections Framework. The class implements Map interfaces including NavigableMap, SortedMap and extends AbstractMap TreeMap in Java does not allow null keys (like Map) and thus a NullPointerException is thrown. However, multiple null values can be associated with different keys. All Map.Entry pairs returned by methods in this class and its views represent snapshots of mappings at the time they were produced. They do not support the Entry.setValue method. It is efficient way of having objects sorted by some key. If also random access is important for you then TreeMap is the answer. With this data structure you can iterate in order. For example the following code :

```
public static void main(String args[]) {
    Map<String, Set<String>> dictionary = new TreeMap<>();
    Set<String> a = new TreeSet<>(Arrays.asList("Actual", "Arrival", "Actuary"));
    Set<String> b = new TreeSet<>(Arrays.asList("Bump", "Bravo", "Basic"));

    dictionary.put("B", b);
    dictionary.put("A", a);

    System.out.println(dictionary);
}
```

Prints out the following code : 

```
{A=[Actual, Actuary, Arrival], B=[Basic, Bravo, Bump]}
```
In the search function is the cached results TreeMap<> does not contain the key called name, then we put this key and the corresponding data to the key in the directory pertaining to the CachingDirectory. You can get the data pertaining to the key then by writing `return cachedResults.get(name)`.

**Practise : part 3**

To check if a map contains null values you can write : `map.values().contains(null)`.

The `putAll(Map<? extends K,? extends V> map)` method is used to copy all of the mappings from the specified map to this map. These mappings replace any mappings that this map had for any of the keys currently in the specified map. But it does not remove the previous data that was there before unless there is data with the same key now.

My rule of a thumb is: if you have to put all key/value pairs from one map to another, then rely on the smartness of the implementor and use the putAll method. There's always a good chance that it provides a better performance than calling put for all pairs manually.

The key difference between HashMap and TreeMap is that HashMap does not maintain a specific order in data elements while TreeMap maintains the ascending order of data elements.

On this document we will be showing a java example on how to use the getOrDefault() method of HashMap Class. Basically this method is to return a default value whenever the value was not found using the key specified on the HashMap. This is a convenient way to handle the scenario where we want a returned other than null which is being returned by get method whenever the key was not found on the HashMap object.

In the search we basically search for the name in the directory. Then we search for it in the overrides where specificities could be specified. If indeed we do have some data in the overrides then we add this data to the results, otherwise we return the default data in the wrapped directory. 

**Practise : part 4**

Functional interfaces are new additions in java 8 which permit exactly one abstract method inside them. These interfaces are also called Single Abstract Method interfaces (SAM Interfaces).

When the input is a command followed by text, then that means that you need to separate the command from the text. But the text can be separated by some spaces hence you should only split some data not all. Java String split () method with regex and length example 2. Here, we are passing split limit as a second argument to this function. This limits the number of splitted strings.

In the case when the command is empty, this means that we have entered nothing in the search bar and we just show the default app view. Otherwise we just have a command and no text and in this case we return a null value. Otherwise when the text is not null then we can call the handleSearch and handleShow methods. The command can only take one of two values. Everysingle time you need to return a new AppView. We need to show the names of the search results, now the person has other data except for their names therefore you need to map each person to its name as follows `map(p -> p.name)`. However before using the map method you need to use the stream method. The collect() method of Stream class can be used to accumulate elements of any Stream into a Collection.

The map() function is a method in the Stream class that represents a functional programming concept. In simple words, the map() is used to transform one object into other by applying a function.

That's why the Stream.map(Function mapper) takes a function as an argument. For example, by using the map() function, you can convert a list of String into a List of Integer by applying the Integer.valueOf() method to each String on the input list. An example is :

```
List<Integer> even = numbers.stream().map(s -> Integer.valueOf(s)).filter(number -> number % 2 == 0).collect(Collectors.toList());
```
The joining() method of Collectors Class, in Java, is used to join various elements of a character or string array into a single string object. This method uses stream to do so. There are various overloads of joining method present in the Collector class. The parameter used in the function is then the delimiter. We therefore can write :

```
results.stream().map(p -> p.name).collect(Collectors.joining(System.lineSeparator()))
```
 
You can get the first element of a list by writing : `list.get(0)`.

**Exam 2018**

**Theory**

The software engineer should put the bug into the backlog. Fuzzing can quickly find difficult corner cases that humans cannot think of. Unit testing requires more manual effort. The Invariant concerns only the class and deals with a variable. The Requires can be used on a method and is a prerequisite for the method to work. Invariants can not concern method. The following erroneous code :

```
Collection<Student> students = ...; // somehow obtain students
Collection<Human> humans = students;
humans.put(new Professor()); // ouch!
```

shows that you can not add other types of humans to a collection of students even when it is cast as a collection of humans. 

**Practise : part 1**

You additionally add a list of Observers. Essentially most of the code was already implemented, we hust throw the corresponding exception when there are specific conditions that are met. We check if the oepration can be performed using `user.canAsk(text)`. Now when you post a question it is not enough to add the question into the array list, the observers should also be notified. There are also two other functions `canAsk` and `canAnswer` when the users can not perform the operations.

Then in the editpost we check whether the questions contains the post or the answers to a question contain the post. With :

```
!questions.contains(post) && questions.stream().noneMatch(q -> q.getAnswers().contains(post))
```

Above here you take each question, the answers related and check whether the post is inside. If none match, meaning that the returned value is true, then we throw the corresponding exception. When you notify the observers what you do essentially is for each observer you take him and run the update function on the observer.

```
observers.forEach(o -> o.update(this, arg));
```

You can make code shorter by checking whether a condition is true such as `return text.length() >= 10;`. It returns a boolean. You can check if an object is an instance of another object as follows with : `post instanceof Question`. Before you cast an object as a specific instance you need to check whether it is an instance of that class.

```
@Override
public boolean equals(Object o) {
    if (!(o instanceof Answer)) {
        return false;
    }
  
    Answer answer = (Answer) o;   // here we cast the object as an answer
    return getAuthor().equals(answer.getAuthor())
            && getText().equals(answer.getText());
}
 ```
 
 **Practise : part 2**
 
In the leaderboard you need to use the observer design pattern. You can use the BigInteger class in the `java.math.BigInteger` library in order to display big integers. The scores are displayed in a HashMap, which is initially empty. How come the leaderboard is acting as an observer of the forum ? Next we update the leaderboard given the post that was added. We have that the arg Object parameter is the post. We cast it as a post. The method `putIfAbsent` probably adds to the scores array when we don't have an entry with the corresponding key `post.getAuthor()`, and it initializes the score to the zero value of the BigInteger class `BigInteger.ZERO`. Then we add the score as follows : `scores.put(post.getAuthor(), scores.get(post.getAuthor()).add(BigInteger.valueOf(score)));`. Here as you can see to convert the long score to a BigInteger you must use the following command : `BigInteger.valueOf(score)`.

When you print the leaderboard you have a rank, so you must originally initlialize it to the zero value. Then to find how many people have the same number of points you need to have an integer called the similar count. When you want to create an ArrayList we need to have the variable initialized as simply a list.

Let's look at an example of Map.Entry

```
import java.util.*;
public class HashMapDemo {

   public static void main(String args[]) {
      // Create a hash map
      HashMap hm = new HashMap();

      // Put elements to the map
      hm.put("Zara", new Double(3434.34));
      hm.put("Mahnaz", new Double(123.22));
      hm.put("Ayan", new Double(1378.00));
      hm.put("Daisy", new Double(99.22));
      hm.put("Qadir", new Double(-19.08));
      
      // Get a set of the entries
      // The entrySet( ) method declared by the Map interface returns a Set containing the map entries. 
      // Each of these set elements is a Map.Entry object.

      Set set = hm.entrySet();
      
      // Get an iterator over this entry set 
      Iterator i = set.iterator();
     
      // Display elements 
      while(i.hasNext()) {
         Map.Entry me = (Map.Entry)i.next();
         System.out.print(me.getKey() + ": ");
         System.out.println(me.getValue());
      }
      System.out.println();
   }
}
```
Since in the below the entrySet returns a set of Map.Entry, then that's why we write that the list is a list of Map.Entry values. 

```
List<Map.Entry<User, BigInteger>> sortedScores = new ArrayList<>(scores.entrySet())
```
The sort() method sorts the elements of a given list in a specific ascending or descending order. In the sort function we call the function which is used here to do the sorting as follows : `list.sort(Leaderboard::compareEntries)`. In the for loop when the values repeat meaning we have the same score then the similarCount increases by own. Else you add the similarCount to the rank, set the similarCount back to one and store in the last variable the current value. To add the line separator we have : `System.lineSeparator()`.

In the sort function we are comparing two entries `compareEntries(Map.Entry<User, BigInteger> a, Map.Entry<User, BigInteger> b)`. The compareTo method works as such. If first string is lexicographically greater than second string, it returns positive number (difference of character value). If first string is less than second string lexicographically, it returns negative number and if first string is lexicographically equal to second string, it returns 0. In our case when `first != 0` that means that the two strings are not equal. Otherwise teh values are equal hence we have the same BigInteger values. We therefore compares the key, and order the keys alphabetically. The compareEntries therefore sorts all the entries on the basis of whether their conjunct value under the function `compareEntries` is negative or positive. 

**Practise : part 3**

In the Limited User class we have the return function depends on two conditions. So you have a double condition as follows : `return isAllowed(text) && user.canEdit(post, text);`. In the class which does the extending you can write `@Override`.

**Exam 2017**

**Theory""

The camelCase notation in Java, which mandates that the first letter of variables and members should be lowercase. 

Code smells are not fatal issues, but may lead to maintainability problems as the codebase evolves. Validation is about determining whether you're building the right thing, and only the customer can decide whether the product is doing the right thing -- by reducing customer involvement (as Waterfall does), you risk correctly building the wrong thing and making the customer unhappy. 

 If we say that a List<Fondue> is a List<Meal>, it means that any code that uses List<Meal> works without modification if we replace the List<Meal> with a List<Fondue>. This is ok for the get method (it would return a Fondue which is indeed a Meal), but it's not ok for the add method: the code would pass a Meal to a method that is expecting a Fondue, but unfortunately a Meal is not a Fondue, so the add call would fail. 
    
The invariant only works on the class, the requires works on the method.

**Practise : part 1**

Why do we have the final keyword for the taught and attended courses list ? In order to sort courses you need to use a comparator. The comparator is used on the class called Course. In this you need to override the compare method. It takes as parameters any two courses and then you need to compare the names of the two courses.

```
final Comparator<Course> coursesSorter = new Comparator<Course>() {

    @Override

    public int compare(Course x, Course y) {

        return x.name.compareTo(y.name);

    }

};
```

Now in order to sort a list of courses, you can also use a comparator as follows :

```
Collections.sort(taughtCourses, coursesSorter);
```

We have three different methods for printing the different parts. In the method you basically return different parameters depending on whether the courses are empty or not. In the for loop you have the keyword final.

**Test Part 1**

Here we have different import statements that we need in order to run tests : 

```
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertThat;
```

Before defining the class with all the tests, you can use the following : `@RunWith(JUnitGradeSheetTestRunner.class)`. In the graded category you may specify the name and the desription of the test class. The first test creates an instance of the Homepage. In the next test we put in between the parentheses the output that is expected from the test which in this case is an IllegalArgumentException.class. You can create an empty collection by writing `Collections.emptyList()`. You can also create a collection with only one element as follows `Collections.singletonList(user)`. You can create an array from a list as follows `Array.asList`. In the test you check that all the methods give the appropriate error or the appropriate output given different input types. Actually you could probably join the methods for sorting the taught and the attended courses into one method no ?

**Practise : part 2**

In the PapademiaUser class, we first throw exceptions given if some variables are null, and if no exception is thrown we initialize the this variables. We have that the taughtCourses is a final variable, and to create such a variable you can write :

```
public final List<PapademiaCourse> taughtCourses; 
this.taughtCourses = Collections.unmodifiableList(new ArrayList<>(taughtCourses));
```

You use the unmodifiableList to create a final variable. The collection is initialized from an ArrayList. Which itself is initialized from the taughtCourses parameter which itself is a final List<PapademiaCourse>. So we have in essence : `public final List<PapademiaCourse> varName = Collections.unmodifiableList(new ArrayList<>(final List<PapademiaCourse> varName))`.

WE have a List<PapademiaUser> but we return a List<User>. To convert this we basically iterate over all the PapademiaUsers then we add a new variable which is a new User which has the name initialized to the name of the corresponding PapademiaUser. If you want to skip a course you write in the if loop a continue to skip the if loop. But when you create a course you not only need the course name but also the lecturers and the students in the course. But now gievn a fixed course name then you need to iterate over all the users to check who among them is a lecturer or a student. If the user teaches the course, then we add a new user with the corresponding lecturer name into the new ArrayList called leturers. 
    
**Practise : part 3**

The getQuirrFormatter always returns a CourseQuizFormatter, and fifferent sub methods are called depending on the type of the user. There you return a new CourseQuizzFormatter where you need to override methods. The parameter of the following equals function is a general object because you should be able to compare the course with any object. Then if the object passed is indeed a course then you can cast the object as a course. You can use the equals() standard method on arraylists and strings but not on classes in general. So you need to create a method to compare classes. 

```

@Override

public boolean equals(Object obj) {

    if (obj == null || !(obj instanceof Course)) {

        return false;

    }

    final Course other = (Course) obj;

    return name.equals(other.name)

            && teachers.equals(other.teachers)

            && students.equals(other.students);

}
```

**Exam 2016**

**Theory**

fill in

**Practise : part 1**

All the classes are declared private static final classes. The method in the class is declared public void. This method can throw an exception. If the function returns an exception then we can write the following code before the function : ` @Test(expected = IllegalArgumentException.class)`. When you want to assert that something is empty you can write : `            assertThat(library.functions.entrySet(), is(empty()));`.

Didn't finish this.

**Exam 2015**

**Practise : part 2**

You can push all the code to github using the following command `git push --all`. First we define the IGraphElementVistor<D>. This interface has three methods called visit that do not return anything but that takes in three different types of input. Next we have the IGraphElement<D> class which has an accept method that takes an IGraphElementVistor<D> type variable as an input parameter. The GraphEdgeIterator implements an Iterator instance which iterates over instances of the GraphEdge<D>. This class takes in a current node, a list of nodes and a current index. It also has a method which is called hasNext. This method returns a boolean which validates whether or not the current index is strictly smaller or not to the whole list od nodes. Meanwhile the next method returns a GraphEdge<D>. If the next GraphEdge exists then we can return a new GraphEdge which depends on the current node and the next node. In the parameters we have the final keyword. We can also have `private final Sale<GraphNode<D>> nodes = new HashSet<>()`. Java 8 has added a method called forEachRemaning to the Iterator interface. This helps in using an Iterator to internally iterate over a Collection, without an explicit loop. We have :
    
```
node.getForwardEdges().forEachRemaining(e -> e.accept(this));
```

**Practise : part 3**

In the NameBook constructor you iterate over different arrays of strings. For a fixed array ofstring, the first element is used to get the corresponding graph node. The second parameter is use to get the next corresponding graph node. In the findFriends method we have check whether either the personNode or the distance is null, if this is the case then we return a collection created from an empty list. When you use the Collection.sort we have as a second parameter a Comparator<String>. You need to create a new instance of this class. In there you need to use the @Override the compare method. 

In the tests we have `import org.junit.Test; import static org.junit.Assert.assertEquals;`. In the class right before the public void method we add the following piece of code `@Test`. Before running the tests, we have the @Before before the method. Now this method throws an IOException. Now we have a text file and we create a new FileInputStream initialized to this text file. We have : `new FileInputStream("friends.txt")`. We can have : `stream().map(GraphNode::getData).collect(Collectors.toList()))` where in the map function we call the getData method in GraphNode. In the collect method we collect the data in a list using Collectors.toList(). 

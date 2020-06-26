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

We can also have assertions, we have : `assert CONDITION : "string to be sent if condition not verified"`. Next we have the value of this character called `ch`, and we find its value, then we find it's index in the string s. We have : `int index = s.indexOf(String.valueOf(ch))`. When this character does not exist in the string we have that the indexOf method returns -1. To initialize an array of integers `private int[] contents = new int[capacity]`. We have contracts in software engineering as follows : requires precondition with @Requires, you can rely on the postcondition with @Ensures. The representation invariants are internal contracts of a class. They must be satisfied after the instantiation of the class and preserved by public class methods.

**Week 4**

Here we can use JUnit as follows : `assertEquals(1+1, 2)`. When you want to use JUnit and Hamcrest then you can write : `assertThat(1+1, is(2))`. You can execute different test codes at different times using the following keywords : `@BeforeAll, @AfterAll, @BeforeEach, @AfterEach`. 

We include here the code to create an array of integers and then take the sum of the individual elements : `sum(new int[] {1,2,3})`. Now here is an example of the use of an interface :

```
interface AuthService {
 User getCurrentUser();
}

// here we are creating a new instance of the interface 
AuthService auth = new AuthService() {
 // inside of this instance we need to override the methods from above
 @Override
 public User getCurrentUser() {
  return null;
 }
}
```

We can also write code to check whether certain calls to methods indeed do return the right exceptions as follows:

```
HttpClient client = new HttpClient() {
 @Override
 public String post(String url, String body) {
  assertThat(body, is("MSG:Hi");
  return "OK";
 }
}

assertThrows(IllegalStateException.class, () -> postMessage("Hi", auth, client))
```

In test-driven development, you first write the tests then write the code. An example where you code the bank accounts on code is as follows :

```
class Account {
 
 private int balance;
 
 int balance() { return balance; }
 
 public Account(int balance) {
  if (balance < 0) {
   throw new IllegalArgumentException();
  }
  
  this.balance = balance;
 }
 
 int withdraw(int amount) {
  int result = Math.min(balance, amount);
  balance -= result;
  return result;
 }
}
```

The corresponding test are :

```
@Test
void partialWithdrawIfBalanceTooLow() {
 Account account = new Account(10);
 assertThat(account.withdraw(20), is(10));
 assertThat(account.balance(), is(0)):
}

@Test
void canWithdrawLessThanBalance() {
 Account account = new Account(100);
 assertThat(account.withdraw(10), is(10));
 assertThat(account.balance(), is(90));
}

@Test
void cannotInitializeWithNegativeBalance() {
 assertThrows(IllegalArgumentException.class, () -> new Account(-1));
}

@Test
void canWithdrawNothing() {
 Account account = new Account(100);
 assertThat(account.withdraw(0), is(0));
}
```

Regression testing is done after the release of the code. In the regression testing you reproduce the bug, then you fix the bug. The coverage notion measures the amount of code executed by tests divided by the total amount of code. The program paths indicate all the possible evolutions of the code and with what sort of output we can be outputted. These can be found by examining the cases when the different boolean methods return different possible values meaning when influential and unpaid bills are respectively equal to true or false. In this code the user can also turn out to be null. The different paths are found with the if statements.

**Week 5**

The waterfall model is defined by the following : requirement specifications, system design, design implementation, verification and test, system deployment, software maintenance. The weakness of the waterfall method is that requirements must be known upfront, the method is inflexible, the customer does not get to review the product. The scrum method has a sprint as a basic structure. A sprint lasts 1-2 weeks, there has to be a working product at the end of each sprint.

In ths scrum method we have inputs from customers, teams and managers to the product owner. The product owner has a product backlog, who creates a sprint planning meeting. Then there is a sprint backlog where all the information relative to the sprint is stored. The product owner represents the end-user's interests. The team builds the product, and it is cross-functional. The team is self-managing, it is autonomous and accountable. In the scrum method there is a scrum master. His role is to ensure the team's success.

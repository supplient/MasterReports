#! https://zhuanlan.zhihu.com/p/404503798
# C++检查模板参数的成员函数是否具有指定签名

如题，仅为个人备忘。该静态检查的效果并不好，和直接依靠编译器进行检查区别不大。

# Problem
我们有伪函数类HashFunc，它有一个成员函数operator()：

``` c++

class HashFunc
{
public:
	size_t operator()(const int& k) {
		return k % 100;
	}
};

```

然后我们还有一个类HashSet，它有一个模板参数：

```c++

template<typename T_HashFunc>
class HashSet
{
public:
	void insert(int val) {
		size_t hashValue = T_HashFunc()(val);
		// ...
	}
	// ...
};

int main() {
	HashSet<HashFunc> s;
	return 0;
}

```

最后我们想要做的是能够在编译阶段就检查T_HashFunc是否是一个我们需要的函数类：
* 有默认构造函数
* 有一个括号重载函数operator()
  * 这个重载函数的返回值是size_t，参数是const int&


# Solution
## 函数签名检查
一步步来，先看怎么检查函数签名。

在c++17中引入了type traits：[is_invocable](https://en.cppreference.com/w/cpp/types/is_invocable)，可以用它来检查一个函数的签名：

```c++
int func(bool& k) {}

static_assert(is_invocable_r_v<int, decltype(&func), bool&>, "Error");
```

上面代码中，func是一个函数名，&func取其函数指针，再用[decltype(&func)](https://en.cppreference.com/w/cpp/language/decltype)取函数签名，最后使用is_invocable检查该签名。具体is_invocable的参数列表请见上面的连接。

## 成员函数指针转普通函数指针
但如果我们直接用上面的方法检查T_HashFunc::operator()的话是不行的，因为它是个成员函数指针，而非普通的函数指针。

为了实现这一目的，我参照了：
* https://stackoverflow.com/questions/56709483/removing-class-pointer-from-member-function-pointer-type
* https://stackoverflow.com/questions/22213523/c11-14-how-to-remove-a-pointer-to-member-from-a-type

就是一个利用template specialization的type traits技巧：
```c++
template<typename>
struct remove_member_pointer {};

template<class T, class U>
struct remove_member_pointer<U T::*> {
	using type = U;
};

using normalFuncPointer = remove_member_pointer<decltype(&HashFunc::operator())>::type;

```

这里using和typedef是类似的东西，是c++11的新语法，可以参阅这里：https://en.cppreference.com/w/cpp/language/type_alias。

对上面代码里的normalFuncPointer使用is_invocable检查函数签名即可。


## 最终实现

```c++
template<typename T_DataType, typename T_HashFunc>
class HashSet
{
	static_assert(
		// Check if is a class, to avoid namespace
		std::is_class_v<T_HashFunc> 
		// Check if constructible
		&& std::is_constructible_v<T_HashFunc>
		// Check operator()'s signature
		&& std::is_invocable_r_v<size_t, remove_member_pointer<decltype(&T_HashFunc::operator())>::type, const T_DataType&>
		// Error message
		, "T_HashFunc must have an overloaded operator() with signature: size_t(const T_DataType&)");
};

class HashFunc
{
public:
	size_t operator()(const int& k) {
		return k % 100;
	}
};

int main() {
	HashSet<int, HashFunc> s;
	return 0;
}
```

# Further Consideration
* 哪怕operator()的返回值不是size_t，而是bool，也照样能过编译，似乎is_invocable并没有那么严格。
* is_invocable也不会严格区分const int&和int。







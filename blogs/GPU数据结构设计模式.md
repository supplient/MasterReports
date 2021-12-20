本文仅仅只是自己在实践中总结的经验。GPGPU编程有很多选择，目前比较流行的有CUDA和OpenCL，本文选择CUDA作为例子。

# 一、松散的数据与接口
## 1. 单核的情况
通常来说，我们设计一个类的时候是将方法和数据打包在一起，这也是OOP的基本思想。让我们从一个简单的CPU端的动态增长的栈入手：

``` c++

// StackInterface<T> is ignored
template<typename T>
class MyStack: public StackInterface<T> {
public:
	MyStack() = default;
	MyStack(const MyStack&) = delete;
	MyStack(const MyStack&& k) {
		if (m_data)
			delete[] m_data;

		m_data = k.m_data;
		m_size = k.m_size;

		k.m_data = nullptr;
	}
	~MyStack() {
		if (m_data)
			delete[] m_data;
	}

public:
	void Push(const T& ele) override {
		EnsureCapacity(m_size + 1);
		m_data[m_size] = ele;
		m_size++;
	}

	T Pop() override {
		assert(m_size > 0);
		const T& ele = m_data[m_size - 1];
		m_size--;
		return move(ele);
	}

	size_t GetSize()const override { return m_size; }

private:
	void EnsureCapacity(size_t newCap) {
		if (newCap <= m_capacity)
			return;
		newCap = m_capacity + max(newCap - m_capacity, m_capacity);
		T* newData = new T[newCap];
		if (m_data) {
			memcpy(newData, m_data, m_size * sizeof(T));
			delete[] m_data;
		}
		m_data = newData;
		m_capacity = newCap;
	}

private:
	T* m_data = nullptr;
	size_t m_size = 0;
	size_t m_capacity = 0;
};

```

它支持Push和Pop这两个常规操作，数据放在CPU上。

那么如果我们希望有一个GPU端的栈呢？可能，我们想在GPU端操作它，所以我们希望它的数据能够放在GPU端，然后接口也暴露在GPU端。就是说，我们可能想写这样的代码：

``` c++ 
__global__ void kernel__UseStack() {
	Stack s;
	s.Push(3);
	s.Push(4);
	s.Pop();
}
```

这可以非常简单，注意到Stack是在一个`__global__`函数中被初始化的，这意味着它包括构造函数、析构函数在内的方法都可以是device function。并且目前CUDA也支持dynamic global memory allocation了，所以简单地写一个：

``` c++
template<typename T>
class MyStack {
public:
	__device__ MyStack() = default;
	__device__ MyStack(const MyStack&) = delete;
	__device__ MyStack(const MyStack&& k) {
		if (m_data)
			delete[] m_data;

		m_data = k.m_data;
		m_size = k.m_size;

		k.m_data = nullptr;
	}
	__device__ ~MyStack() {
		if (m_data)
			free(m_data);
	}

public:
	__device__ void Push(const T& ele) {
		EnsureCapacity(m_size + 1);
		m_data[m_size] = ele;
		m_size++;
	}

	__device__ T Pop() {
		assert(m_size > 0);
		const T& ele = m_data[m_size - 1];
		m_size--;
		return move(ele);
	}

	__device__ size_t GetSize()const { return m_size; }

private:
	__device__ void EnsureCapacity(size_t newCap) {
		if (newCap <= m_capacity)
			return;
		newCap = m_capacity + max(newCap - m_capacity, m_capacity);
		T* newData = static_cast<T*>(malloc(sizeof(T) * newCap));
		if (m_data) {
			memcpy(newData, m_data, m_size * sizeof(T));
			free(m_data);
		}
		m_data = newData;
		m_capacity = newCap;
	}

private:
	T* m_data = nullptr;
	size_t m_size = 0;
	size_t m_capacity = 0;
};

```

改动基本就只有追加了``__device__`` annotation而已。



## 2. 需要GPU多线程共享的情况

上述两个Stack的实现之所以这么方便，是因为它们的数据和所有接口都是统一在一个地方的：第一个Stack是在CPU端，第二个Stack是在GPU端的一个线程内。

我们可能会希望能够在GPU的多个线程之间共享这个栈，这一方面会带来同步问题，不过那不是这篇文章的重点，所以先忽略它，另一方面是内存管理的问题。虽然device malloc分配的global memory在多个线程之间是共享的，但这个memory的指针却是各个线程的local variable，更何况我们希望只有一个被共享的栈，而不是每个线程都分配一个。

所以一个直接的想法是让launch kernel之前，从CPU端分配一个驻留在GPU端的栈。使用起来大概是这种感觉：

``` c++
__global__ void kernel__Init(Stack* s) {
	// placement new
	s = new(s) Stack();
}
__global__ void kernel__Test(Stack* s) {
	s.push(threadIdx.x);
}

void Test() {
	Stack* d_s;
	cudaMalloc(&d_s, sizeof(Stack));
	kernel__Init<<<1,1>>>(d_s);
	kernel__Test<<<1,4>>>(d_s);
}
```

值得注意的地方是我们需要在使用之前先launch一个用于调用构造函数的kernel(`kernel__init`)，毕竟我们没有使用方便好用的new operator来分配内存，构造函数是不会被自动调用的。

这样可以解决问题，让多个线程共享一个栈，但是毕竟每次都要手动launch kernel__init实在是麻烦，所以我们稍微给它封装一下：

``` c++
namespace kernel {
	template<typename T, typename... T_Args>
	__global__ void ConstructObject(T* obj, T_Args... args) {
		obj = new(obj) T(args...);
	}
	template<typename T>
	__global__ void DestructObject(T* obj) {
		obj->~T();
	}
}

template<typename T>
class HostWrap {
public:
	HostWrap() {
		cudaMalloc(&m_d_obj, sizeof(T));
		kernel::ConstructObject<T> << <1, 1 >> > (m_d_obj);
	}
	~HostWrap() {
		kernel::DestructObject<T> << <1, 1 >> > (m_d_obj);
		cudaFree(m_d_obj);
	}
	
	operator T*()const { return m_d_obj; }

private:
	T* m_d_obj;
};

namespace kernel {
	template<typename T_Stack>
	__global__ void Test(T_Stack* s, size_t n) {
		size_t ti = threadIdx.x;
		
		// Push
		for (size_t i = 0; i < n; i++) {
			if (ti == i)
				s->Push(ti);
			__syncthreads();
		}

		// Pop
		for (size_t i = 0; i < n; i++) {
			if (ti == i)
				assert(s->Pop() == n - 1 - i);
			__syncthreads();
		}
	}
}

template<typename T>
class Test {
public:
	void operator()() {
		HostWrap<T> stackWrap;
		kernel::Test<T> << <1, 4 >> > (stackWrap, 4);
		cudaDeviceSynchronize();
	}
};

int main() {
	Test<MyStack<int>>()();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}

```

封装的代码里还包含了调用析构函数的代码，这也是显然的，毕竟`cudaFree`并不会自动调用析构函数。



## 3. 不使用device malloc的情况
现在我们需要非常大的栈，大到可能会有1GB那么大的程度。





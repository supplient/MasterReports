#! https://zhuanlan.zhihu.com/p/418985780
# CUDA中原子锁的实现

# 问题：一个需要锁的场景
假设我有这么一个函数需要若干线程并行执行（也就是临界区critical section）：
```c++
__device__ void device_TestFunc(uint32_t ti, size_t* value) {
	printf("Thread %i entered with value %i\n", (int32_t)ti, (int32_t)*value);
	*value = *value + 1;
	printf("Thread %i exited with value %i\n", (int32_t)ti, (int32_t)*value);
}
```
就是自增value指向的值，其实最好是直接使用atomicInc或者atomicAdd这些原子操作函数的，不过我们这边为了演示，还是自己手动实现一个原子锁。

如果不上锁的话，kernel就大概是这么写（host端的代码最后再一起放出来）：
```c++
__global__ void kernel__Test_origin(size_t n, size_t* value) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	device_TestFunc(ti, value);
}
```

此时因为GPU的执行是SIMD的，所以函数`device_TestFunc`中的代码都被一条条同时执行：大家都先读`*value`的值，再加一，最后把值存回去。结果就导致这样的输出：
```
---- Test Origin ----
Thread 0 entered with value 0
Thread 1 entered with value 0
Thread 2 entered with value 0
Thread 3 entered with value 0
Thread 4 entered with value 0
Thread 0 exited with value 1
Thread 1 exited with value 1
Thread 2 exited with value 1
Thread 3 exited with value 1
Thread 4 exited with value 1
```
可以看到，所有线程读到的旧值都是0，写入的新值都是1。


# 简单的尝试：SpinLock
如果熟悉多线程编程的话，可能会直接上手写个[自旋锁](https://en.wikipedia.org/wiki/Spinlock)：
```c++
// Initially, *lock == 1
__global__ void kernel__Test_SpinLock(size_t n, size_t* value, int32_t* lock) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	// Lock
	while (atomicExch(lock, 0) == 0)
		;
	// Work
	device_TestFunc(ti, value);
	// Unlock
	*lock = 1;
}
```
这就是让各个线程轮流进入Work区域，如果还没轮到的话就在Lock区域的while循环处忙等待。

这一实现在MIMD的多核CPU中是没问题的，但是在SIMD的GPU中是不可行的：因为一个wrap内的线程总是完完全全SIMD的，只要有一个线程还在Lock区域等待，这一整个wrap内的线程都不会继续往下执行。这就导致拿到了lock的那个线程也被迫等待在Lock区域，而无法进一步执行到Work区域，然后在Unlock区域释放lock。

结果是会死锁：拿到了lock的A线程等待没有拿到lock的B线程继续向下执行，但是B线程也同时在等待A线程释放lock。

这一问题在nvidia的论坛里也被问过：https://forums.developer.nvidia.com/t/try-to-use-lock-and-unlock-in-cuda/50761 


# 针对GPU的修复方法：TicketLock
这一节的实现参见了《CUDA by Example》的A.2.4。

注意到其实只有wrap内部是严格SIMD的，而wrap之间是可以不用在同一时间执行同一条指令的，换句话说在wrap之间是可以用上一节的自旋锁的。所以只要消除wrap内部的死锁就行了。

解决思路的基础是wrap大小是固定的32。我们给wrap内的每个线程编号：0, 1, 2, ..., 31。然后用一个循环来给每个线程一次拿到lock的机会：
``` c++
__global__ void kernel__Test_TicketLock(size_t n, size_t* value, int32_t* lock) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	// For each thread in a wrap
	for (int i = 0; i < 32; i++) {
		// Check if it is this thread's turn
		if (ti % 32 != i)
			continue;

		// Lock
		while (atomicExch(lock, 0) == 0)
			;
		// Work
		device_TestFunc(ti, value);
		// Unlock
		*lock = 1;
	}
}
```
这个实现和[TicketLock](https://en.wikipedia.org/wiki/Ticket_lock)差不多。

它之所以不会死锁是因为没有拿到锁的线程们是会继续向下执行的，只是它们不会实际执行Lock-Work-UnLock这块代码，而是nop它们（就是虽然处理器的pc走过这些指令了，但并没有实际的decode, execute等过程），这就相当于发生一次控制分支。循环展开的话大概是这种感觉：
``` c++
if(ti == 0) {
	// Thread 0 will enter here
	// Other threads will nop these code but still execute them

	// Lock
	while (atomicExch(lock, 0) == 0)
		;
	// Work
	device_TestFunc(ti, value);
	// Unlock
	*lock = 1;
}

if(ti == 1) {
	// Thread 1 will enter here
	// Other threads will nop these code but still execute them

	// Lock
	while (atomicExch(lock, 0) == 0)
		;
	// Work
	device_TestFunc(ti, value);
	// Unlock
	*lock = 1;
}

// ......
```



# 我设计的更加一般化的实现：LoopLock
LoopLock是我自己起的名字。

上一节中的TicketLock已经足够解决问题。它的性能问题当然还是一个大问题，同样的代码得要执行32遍呢，不过这个问题除非是改成使用lock-free的实现（例如使用原子函数atomicAdd），不然是解决不了的，毕竟一个wrap内SIMD是目前GPU的特性。

所以这一节我想解决的问题不是性能问题，而是消除掉上一节中的`32`这个数字，毕竟把wrap数量写死在代码里并不是太优雅的事情。

注意到TicketLock解决问题的关键是避免了“拿到锁的线程原地等待没有拿到锁的线程，无法进入临界区”的情况。而要避免这一情况就是要引入控制分支，这一分支中，拿到锁的线程进入临界区，没有拿到锁的线程什么也不做。有了这个思路后，接下来的实现就呼之欲出了：
``` c++
__global__ void kernel__Test_LoopLock(size_t n, size_t* value, int32_t* lock) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	// Loop for introducing control divergence
	while(true) {
		// Try - Lock
		if(atomicExch(lock, 0) != 0) {
			// Work
			device_TestFunc(ti, value);
			// Unlock
			*lock = 1;
			break;
		}
	}
}
```
LoopLock的执行和TicketLock是一模一样的（有可能性能会差点吧，毕竟这里是dynamic loop，而TicketLoop是可以被循环展开的），只是写法上不同而已，这里我们就不用写死`32`这个数字了，各个线程的编号将由GPU对atomicExch的调度来决定。



# 源代码
这一节没别的，就是代码。封装了的LoopLock的代码和实验用的代码。

## 封装了的LoopLock
``` c++
class LoopLock{
private:
	using LockType = int32_t;

public:
	class GPUVer {
	public:
		GPUVer(LockType* d_lock): d_lock(d_lock) {}

		template<typename T_Func>
		__device__ void WithLock(T_Func func) {
			do {
				if (atomicCAS(d_lock, 1, 0) == 1) {
					func();
					*d_lock = 1;
					break;
				}
			} while (true);
		}

	private:
		LockType* d_lock;
	};

public:
	cudaError_t Alloc() {
		cudaError_t et;

		if ((et = cudaMalloc(reinterpret_cast<void**>(&d_lock), sizeof(LockType))) != cudaSuccess)
			return et;
		LockType lock = 1;
		if ((et = cudaMemcpy(d_lock, &lock, sizeof(LockType), cudaMemcpyHostToDevice)) != cudaSuccess)
			return et;

		if ((et = cudaMalloc(reinterpret_cast<void**>(&d_gpuVer), sizeof(GPUVer))) != cudaSuccess)
			return et;
		GPUVer gpuVer(d_lock);
		if ((et = cudaMemcpy(d_gpuVer, &gpuVer, sizeof(GPUVer), cudaMemcpyHostToDevice)) != cudaSuccess)
			return et;

		return et;
	}
	cudaError_t Free() {
		cudaError_t et;
		if ((et = cudaFree(d_lock)) != cudaSuccess)
			return et;
		d_lock = nullptr;
		if ((et = cudaFree(d_gpuVer)) != cudaSuccess)
			return et;
		d_gpuVer = nullptr;
		return et;
	}

private:
	// 1: not locked
	// 0: locked
	LockType* d_lock = nullptr;

public:
	GPUVer* GetGPUVer() {
		return d_gpuVer;
	}

private:
	GPUVer* d_gpuVer = nullptr;
};
```


## 实验用的全部代码
``` c++
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
using namespace std;

#define CheckCuda(x) {\
	if(x != cudaSuccess) \
		cerr << "[CUDA Error][" << __FILE__ << " Line " << __LINE__ << "]\n\t[" \
			<< cudaGetErrorName(x) << "] " \
			<< cudaGetErrorString(x) << endl; \
}


class LoopLock{
private:
	using LockType = int32_t;

public:
	class GPUVer {
	public:
		GPUVer(LockType* d_lock): d_lock(d_lock) {}

		template<typename T_Func>
		__device__ void WithLock(T_Func func) {
			do {
				if (atomicCAS(d_lock, 1, 0) == 1) {
					func();
					*d_lock = 1;
					break;
				}
			} while (true);
		}

	private:
		LockType* d_lock;
	};

public:
	cudaError_t Alloc() {
		cudaError_t et;

		if ((et = cudaMalloc(reinterpret_cast<void**>(&d_lock), sizeof(LockType))) != cudaSuccess)
			return et;
		LockType lock = 1;
		if ((et = cudaMemcpy(d_lock, &lock, sizeof(LockType), cudaMemcpyHostToDevice)) != cudaSuccess)
			return et;

		if ((et = cudaMalloc(reinterpret_cast<void**>(&d_gpuVer), sizeof(GPUVer))) != cudaSuccess)
			return et;
		GPUVer gpuVer(d_lock);
		if ((et = cudaMemcpy(d_gpuVer, &gpuVer, sizeof(GPUVer), cudaMemcpyHostToDevice)) != cudaSuccess)
			return et;

		return et;
	}
	cudaError_t Free() {
		cudaError_t et;
		if ((et = cudaFree(d_lock)) != cudaSuccess)
			return et;
		d_lock = nullptr;
		if ((et = cudaFree(d_gpuVer)) != cudaSuccess)
			return et;
		d_gpuVer = nullptr;
		return et;
	}

private:
	// 1: not locked
	// 0: locked
	LockType* d_lock = nullptr;

public:
	GPUVer* GetGPUVer() {
		return d_gpuVer;
	}

private:
	GPUVer* d_gpuVer = nullptr;
};


__device__ void device_TestFunc(uint32_t ti, size_t* value) {
	printf("Thread %i entered with value %i\n", (int32_t)ti, (int32_t)*value);
	*value = *value + 1;
	printf("Thread %i exited with value %i\n", (int32_t)ti, (int32_t)*value);
}
static constexpr size_t N = 5;


// =====================================================
//	Origin
// =====================================================
__global__ void kernel__Test_origin(size_t n, size_t* value) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	device_TestFunc(ti, value);
}

void Test_origin() {
	size_t* d_value;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_value), sizeof(size_t)));
	size_t value = 0;
	CheckCuda(cudaMemcpy(d_value, &value, sizeof(size_t), cudaMemcpyHostToDevice));

	kernel__Test_origin<<<1, N>>>(N, d_value);
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(d_value));
}




// =====================================================
//	LoopLock Class
// =====================================================
__global__ void kernel__Test_LoopLock_Class(size_t n, size_t* value, LoopLock::GPUVer* lock) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	lock->WithLock(
		[&]() {
			device_TestFunc(ti, value);
		}
	);
}

void Test_LoopLock_Class() {
	LoopLock lock;
	CheckCuda(lock.Alloc());

	size_t* d_value;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_value), sizeof(size_t)));
	size_t value = 0;
	CheckCuda(cudaMemcpy(d_value, &value, sizeof(size_t), cudaMemcpyHostToDevice));

	kernel__Test_LoopLock_Class<<<1, N>>>(N, d_value, lock.GetGPUVer());
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(d_value));
	CheckCuda(lock.Free());
}






// =====================================================
//	LoopLock
// =====================================================
__global__ void kernel__Test_LoopLock(size_t n, size_t* value, int32_t* lock) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	// Loop for introducing control divergence
	while(true) {
		// Try - Lock
		if(atomicExch(lock, 0) != 0) {
			// Work
			device_TestFunc(ti, value);
			// Unlock
			*lock = 1;
			break;
		}
	}
}

void Test_LoopLock() {
	int32_t* d_lock;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_lock), sizeof(int32_t)));
	int32_t lock = 1;
	CheckCuda(cudaMemcpy(d_lock, &lock, sizeof(int32_t), cudaMemcpyHostToDevice));

	size_t* d_value;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_value), sizeof(size_t)));
	size_t value = 0;
	CheckCuda(cudaMemcpy(d_value, &value, sizeof(size_t), cudaMemcpyHostToDevice));

	kernel__Test_LoopLock<<<1, N>>>(N, d_value, d_lock);
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(d_value));
	CheckCuda(cudaFree(d_lock));
}





// =====================================================
//	TicketLock
// =====================================================
__global__ void kernel__Test_TicketLock(size_t n, size_t* value, int32_t* lock) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	for (int i = 0; i < 32; i++) {
		if (ti % 32 != i)
			continue;

		while (atomicExch(lock, 0) == 0)
			;
		device_TestFunc(ti, value);
		*lock = 1;
	}
}

void Test_TicketLock() {
	int32_t* d_lock;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_lock), sizeof(int32_t)));
	int32_t lock = 1;
	CheckCuda(cudaMemcpy(d_lock, &lock, sizeof(int32_t), cudaMemcpyHostToDevice));

	size_t* d_value;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_value), sizeof(size_t)));
	size_t value = 0;
	CheckCuda(cudaMemcpy(d_value, &value, sizeof(size_t), cudaMemcpyHostToDevice));

	kernel__Test_TicketLock<<<1, N>>>(N, d_value, d_lock);
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(d_value));
	CheckCuda(cudaFree(d_lock));
}




// =====================================================
//	SpinLock
// =====================================================
__global__ void kernel__Test_SpinLock(size_t n, size_t* value, int32_t* lock) {
	auto ti = blockDim.x * blockIdx.x + threadIdx.x;
	if (ti >= n)
		return;
	while (atomicExch(lock, 0) == 0)
		;
	device_TestFunc(ti, value);
	*lock = 1;
}

void Test_SpinLock() {
	int32_t* d_lock;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_lock), sizeof(int32_t)));
	int32_t lock = 1;
	CheckCuda(cudaMemcpy(d_lock, &lock, sizeof(int32_t), cudaMemcpyHostToDevice));

	size_t* d_value;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_value), sizeof(size_t)));
	size_t value = 0;
	CheckCuda(cudaMemcpy(d_value, &value, sizeof(size_t), cudaMemcpyHostToDevice));

	kernel__Test_SpinLock<<<1, N>>>(N, d_value, d_lock);
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(d_value));
	CheckCuda(cudaFree(d_lock));
}




int main()
{
	cout << "---- Test Origin ----\n";
	Test_origin();
	cout << "\n";

	cout << "---- Test LoopLock Class ----\n";
	Test_LoopLock_Class();
	cout << "\n";

	cout << "---- Test LoopLock ----\n";
	Test_LoopLock();
	cout << "\n";

	cout << "---- Test TicketLock ----\n";
	Test_TicketLock();
	cout << "\n";

	cout << "---- Test SpinLock ----\n";
	Test_SpinLock();
	cout << "\n";
	return 0;
}
````

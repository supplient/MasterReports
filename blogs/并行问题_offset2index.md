#! https://zhuanlan.zhihu.com/p/430216738
# 并行算法：反向scan，将片段偏移量转化为片段索引

# Problem
* Input: A positive integer array $A = \{a_1, a_2, \dotsc, a_n\}$
* Output: An array $B = \{\overbrace{1, 1, \dotsc}^{a_1}, \overbrace{2, 2, \dotsc}^{a_2}, \dotsc ,\overbrace{n, n, \dotsc}^{a_n} \}$
* Time complexity requirement: $O(\sum_1^n a_i)$

If there are $\sum_1^n a_i$ processors, the algorithm should use only $O(1)$ time.

e.g.

* IN: $A=\{3,5\}$
* OUT: $B=\{1,1,1,2,2,2,2,2\}$


# Solution

``` c++

// inclusive_scan is too complex, so be ignored
// can refer to https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

void memset(int* B, size_t m, int value) {
	for each thread i in [0,m):
		B[i] = value;
}

void set_mark(int* B, int* A, size_t n) {
	for each thread i in [0,n-1):
		B[A[i]] = 1;
}

int* Algo(int* A, size_t n) {
	// Input
	// A={3, 5}, n=2

	inclusive_scan(A, n);
	// A={3, 8}
	// O(n)

	size_t m = A[n-1];
	// m=8
	// O(1)

	int* B = new int[m];
	memset(B, m, 0);
	// B={0,0,0,0,0,0,0,0}
	// O(m), ignore memory allocation cost

	set_mark(B, A, n);
	// B={0,0,0,1,0,0,0,0}
	// O(n)

	inclusive_scan(B, m);
	// B={0,0,0,1,1,1,1,1}
	// O(m)

	// Total: O(m+n) = O(m)

	return B;
}
```


# Benchmark
refer to gist: https://gist.github.com/supplient/cf657df60332a198992602f6452fd532
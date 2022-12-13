在课程中，我们使用了一个 CUDA 示例来相加两个一维数组(即向量)。具体来说，我们将一个大小为N的数组与另一个数组相加

`C[N] = a[N] + b[N]`

你的课堂活动是为了演示在 GPU 卡上并行计算的基本步骤。在本作业中，将扩展课堂活动，以包括线程之间的合作，更好地了解 CPU 和 GPU 的性能与操作和数据大小的关系。

1. 使用共享目录下的代码样本 (`gpu_vector_add.cu`, `timer_nv.h`) 来学习如何为 CPU 函数和 GPU 内核计时。为内核计时与为在 CPU 上运行的函数计时是不同的。使用 CPU 定时器(`timer_nv.h`)来为 CPU 函数计时，而用 GPU 定时器为 GPU 内核计时。你应该只对内核进行计时，并且是包括内存分配和从 GPU 来回复制数据的代码部分进行计时，以了解在GPU上移动数据与数字运算的成本。

2. 测量CPU函数和GPU内核的以下大小的向量加法例子的时间，并以表格形式展示 你的时间。你可以用MS Word来记录你的结果。


```
  a.1,000,000
    vector_add on the CPU. z[100] = 5.00
    elapsed wall time (CPU) = 3.8490 ms
    vector_add on the GPU. z[100] = 5.00
    elapsed wall time (GPU) = 2.7258 ms
  b.10,000,000 
    vector_add on the CPU. z[100] = 5.00
    elapsed wall time (CPU) = 39.0720 ms
    vector_add on the GPU. z[100] = 5.00
    elapsed wall time (GPU) = 25.5650 ms
  c.100,000,000
    vector_add on the CPU. z[100] = 5.00
    elapsed wall time (CPU) = 480.6920 ms
    ector_add on the GPU. z[100] = 5.00
    elapsed wall time (GPU) = 374.8930 ms
```

3. 修改相加矢量的例子，创建一个CPU函数和一个GPU函数，用RADIUS 3元素进行 模版计算(stencil computation)(见讲座幻灯片)。你需要为你的GPU内核创建一个全局内存和共享内存版本。共享内存的实现是比较复杂的。请参考CUDA讲座幻 灯片中的问题陈述。注意，在这个练习中你需要使用cudaMallocManaged函数，而 不是在GPU上显式分配内存的老方法。在使用cudaMallocManaged时，避免为1个区 块和1个线程启动一个内核(即myKernel<<<1,1>>(...) )注意，在使用 cudaMallocManaged时，你需要在内核之后使用cudaDeviceSynchronize()。原因是有了统一内存，我们不再使用cudaMemCpy，而cudaMemCpy是充当同步器的。测量CPU函数和 GPU内核对以下大小的向量进行模版计算的时间，并以表格的形式展示你的时间
。
```
   a.1,000,000
    CPU Execution Time = 0.694622
    GPU Execution Time = 0.139648 (GlobalMem)
    GPU Execution Time = 0.096416 (SharedMem)
   b.10,000,000
    CPU Execution Time = 5.73324
    GPU Execution Time = 0.71648 (GlobalMem)
    GPU Execution Time = 0.13442 (SharedMem)
   c.100,000,000
    CPU Execution Time = 67.43144
    GPU Execution Time =  1.39744 (GlobalMem)
    GPU Execution Time =  0.12323 (SharedMem)
```

4. 重复任务3，但RADIUS=5,7,9,11 观察相同矢量大小下的执行时间变化。观察CPU 和GPU在同样大小的数据下，矢量加法与模版计算的执行时间变化。对于 N=常数， 你认为模版计算或者矢量加法是常量吗?改变RADIUS如何影响CPU和GPU的性 能?通过比较全局内存和共享内存的实现，研究RADIUS的大小对GPU实现的影 响。对你的发现进行总结。

RADIUS 增大、GPU 相对于CPU的加速就越明显、原因是vector add 是访存瓶颈而不是计算瓶颈、 增加计算的密集度、可以平衡计算和访问之间的性能、相对cpu的加速也越高。
RADIUS 增大、使用 sharedMem 的效果也就越明显、原因是对sharedMEM中的数据的重复利用率提高了、从sharedMem中读取数据的速度是globalmem的10倍左右、因此性能提升更加明显。

5. 对于100,000,000的向量大小，在GPU上的全局内存和共享内存实现中，测试每个块的线程数:8、16、32、64、128、256、512、1024，并比较执行时间。对你的发 现发表评论。请注意，你应该只测量内核的时间。

```
8
GPU Execution Time = 0.513184 (GlobalMem)
GPU Execution Time = 0.340096 (SharedMem) 
16
GPU Execution Time = 0.266688 (GlobalMem)
GPU Execution Time = 0.180416 (SharedMem) 
32
GPU Execution Time = 0.158304 (GlobalMem)
GPU Execution Time = 0.095744 (SharedMem)
64
GPU Execution Time = 0.145152 (GlobalMem)
GPU Execution Time = 0.090720 (SharedMem)
128
GPU Execution Time = 0.144288 (GlobalMem)
GPU Execution Time = 0.097248 (SharedMem)
256
GPU Execution Time = 0.143328 (GlobalMem)
GPU Execution Time = 0.095584 (SharedMem) 
512
GPU Execution Time = 0.141472 (GlobalMem)
GPU Execution Time = 0.097152 (SharedMem)
1024
GPU Execution Time = 0.144736 (GlobalMem)
GPU Execution Time = 0.103712 (SharedMem)
```

warp size 是 32 所以 BLOCK_SIZE 应为32 的倍数取得较好的性能8和16的性能较差
warp size 为32倍数时、在保证发射更多的线程的同时、还应该保证SM有足够的活跃度、因此对于该case 后面几个配置性能基本维持稳定


6. 使用你的 CPU 代码进行最大 N 的向量加法和模版计算。 报告调试模式的时间(使 用标志-g)优化级别-O2和-O3(字母O)。

makefile 中增加 flag 对比时间、编译器对 CPU 代码进行了优化
O0  : 0.694584
O2 : 0.000001
O3 : 0.000001

7. 遵循作业提交说明。使用 a2ps 将任务2和3的代码以pdf文件的形式提交。请确保提供 CPU 和 GPU 的型号、CPU 的时钟速度、缓存大小和计算机上的 DRAM 类型。(提示:可以使用 `nvidia-smi` 命令了解该节点上的 GPU, CPU 的相关信息可以查看文件 `/proc/cpuinfo` 和 `/proc/meminfo`。

    A : 环境说明

    硬件：

    - GPU : Tesla V100-SXM2 16GB (nvidia-smi)
    - CPU : Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

        cpu MHz : 2400

        cache size : 28160 KB  
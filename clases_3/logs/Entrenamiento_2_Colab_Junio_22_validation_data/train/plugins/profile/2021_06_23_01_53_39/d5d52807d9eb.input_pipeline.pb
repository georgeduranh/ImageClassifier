	3P??\@3P??\@!3P??\@	"d?????"d?????!"d?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL3P??\@??-Y?=@1a???|?T@A?9τ&??I???w??Y?MbX9??rEagerKernelExecute 0*	??v??f@2U
Iterator::Model::ParallelMapV2jm?kA??!??3q??9@)jm?kA??1??3q??9@:Preprocessing2F
Iterator::Modely?P?????!?U??B?H@)?a?1????1?۞ԚC8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?x?&1??!?[?&(6@)?+?j???1eh?5?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate$?@??!K?.?5@)Z_&????1?[x??a-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????#??!????K?@)????#??1????K?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?	??ϛ??!X?]?I@)?????z?1
!?~?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?i2?m?w?!}Ij%?'
@)?i2?m?w?1}Ij%?'
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_	?Į???!??#A8@)U2 Tq?v?1??)??P	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 26.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9"d?????I??i;@Q?(Lm#R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??-Y?=@??-Y?=@!??-Y?=@      ??!       "	a???|?T@a???|?T@!a???|?T@*      ??!       2	?9τ&???9τ&??!?9τ&??:	???w?????w??!???w??B      ??!       J	?MbX9???MbX9??!?MbX9??R      ??!       Z	?MbX9???MbX9??!?MbX9??b      ??!       JGPUY"d?????b q??i;@y?(Lm#R@
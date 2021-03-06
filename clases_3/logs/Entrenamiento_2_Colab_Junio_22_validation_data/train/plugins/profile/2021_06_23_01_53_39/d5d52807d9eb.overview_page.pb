?	3P??\@3P??\@!3P??\@	"d?????"d?????!"d?????"?
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
	??-Y?=@??-Y?=@!??-Y?=@      ??!       "	a???|?T@a???|?T@!a???|?T@*      ??!       2	?9τ&???9τ&??!?9τ&??:	???w?????w??!???w??B      ??!       J	?MbX9???MbX9??!?MbX9??R      ??!       Z	?MbX9???MbX9??!?MbX9??b      ??!       JGPUY"d?????b q??i;@y?(Lm#R@?"h
<gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterS?Wa?	??!S?Wa?	??0"9
sequential_2/Conv2/Relu_FusedConv2D|3?HIr??!r?n???"f
;gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Ah9???!k?5Ql???0"h
<gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter͘??~???!?(VL$??0"f
;gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropInputConv2DBackpropInput+???Ͱ?!TP??ɫ??0"9
sequential_2/Conv4/Relu_FusedConv2Dqk?????!"??????"h
<gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???!???!Ԓ??????0"9
sequential_2/Conv3/Relu_FusedConv2D???	???!?%?K??"f
;gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?}??1o??!?fn?>`??0"h
<gradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?+	c
??!?????P??0Q      Y@Y     @/@a     U@q?>TL?+@y??O??t?"?

both?Your program is POTENTIALLY input-bound because 26.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?14.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 
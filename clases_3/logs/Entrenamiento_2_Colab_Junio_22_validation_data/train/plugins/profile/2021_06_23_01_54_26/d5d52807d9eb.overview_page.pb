?	??=`@??=`@!??=`@	?Y?ɔ???Y?ɔ??!?Y?ɔ??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??=`@?????@1???[?^@A?V??????I??k?????Y?GĔH???rEagerKernelExecute 0*	}?5^?)_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatjj?Z_??!??<????@)Z+??6??1?_[W?:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?i?*???!B?|*?PA@)1y?|??1?j}?,O8@:Preprocessing2F
Iterator::Model???)r??!!
??9@)%?z?ۡ??1??ꭗ?+@:Preprocessing2U
Iterator::Model::ParallelMapV2??(?????!?ROX??'@)??(?????1?ROX??'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???X??!?+?7?$@)???X??1?+?7?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ݒ???!x?8?ЎR@):̗`}?1?-g???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor8????Cy?!?˅???@)8????Cy?1?˅???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?^I?\??!?+FƨMB@)??Z?a/d?1?s-y????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Y?ɔ??I ?n2?P@Q&?`?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?????@!?????@      ??!       "	???[?^@???[?^@!???[?^@*      ??!       2	?V???????V??????!?V??????:	??k???????k?????!??k?????B      ??!       J	?GĔH????GĔH???!?GĔH???R      ??!       Z	?GĔH????GĔH???!?GĔH???b      ??!       JGPUY?Y?ɔ??b q ?n2?P@y&?`?W@?"h
<gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,"???!,"???0"h
<gradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?L????!?s&[?b??0"f
;gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?a?EͶ?!;RFt?d??0"9
sequential_2/Conv2/Relu_FusedConv2D??+p8??!eJ{?2??"f
;gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropInputConv2DBackpropInput??;ig??!k+?ɵL??0"h
<gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb??????!??O??(??0"h
<gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???@ɳ??!??\7???0"9
sequential_2/Conv3/Relu_FusedConv2D?8?F?^??!$??;????"9
sequential_2/Conv4/Relu_FusedConv2D\?O?(???!ڂʋ	??"f
;gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropInputConv2DBackpropInput??????!??j%X[??0Q      Y@Y???/@a??U@q?V?????yn?ъ?.j?"?	
both?Your program is POTENTIALLY input-bound because 4.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 
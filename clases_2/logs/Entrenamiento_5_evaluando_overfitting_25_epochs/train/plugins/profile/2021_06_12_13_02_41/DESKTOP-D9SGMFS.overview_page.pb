?	??H.?!P@??H.?!P@!??H.?!P@	o Il}??o Il}??!o Il}??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??H.?!P@???h o??A?z6?P@Y?B?i?q??rEagerKernelExecute 0*	     ?z@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?J?4??!?(f>G@)K?46??1b???i?B@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$(~????!r?	<B@)V}??b??1<??j@@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat+??????!????RO"@)??ܵ?|??1.r?	<@:Preprocessing2U
Iterator::Model::ParallelMapV2lxz?,C??!K=?]?@)lxz?,C??1K=?]?@:Preprocessing2F
Iterator::ModelR???Q??!?^m??L&@)8??d?`??1???cy?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8??d?`??!???cy?@)2??%䃎?1*?F1?@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?I+???!??NC?@)?I+???1??NC?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?z6?>??!f>??PE@)ŏ1w-!?1<??j???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range_?Q?{?!??PI7???)_?Q?{?1??PI7???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!E?3????){?G?zt?1E?3????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceU???N@s?!7?ٟ???)U???N@s?17?ٟ???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice_?Q?[?!??PI7???)_?Q?[?1??PI7???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9o Il}??I?}?I??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???h o?????h o??!???h o??      ??!       "      ??!       *      ??!       2	?z6?P@?z6?P@!?z6?P@:      ??!       B      ??!       J	?B?i?q???B?i?q??!?B?i?q??R      ??!       Z	?B?i?q???B?i?q??!?B?i?q??b      ??!       JCPU_ONLYYo Il}??b q?}?I??X@Y      Y@q???????"?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 
?  *	???̌??@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?? ޢ"@!??E?r'V@)???<,?"@1ģ???V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??_?L??!??????%@)?0?*??1z쫙??%@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatO??e?c??!??<??w??)???<,Ԛ?1;?t?????:Preprocessing2F
Iterator::Model??6???!?IX?~??)?0?*???1tp?=?w??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???&??!??d?q???)-C??6??1??T??)??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch	?^)ˀ?!?5j)????)	?^)ˀ?1?5j)????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipGr?????!h?v@&@)???Q?~?1???kB??:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1?|?!?F???
??)y?&1?|?1?F???
??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??0?*x?!!??k???)??0?*x?1!??k???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceŏ1w-!o?!>*????)ŏ1w-!o?1>*????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range{?G?zd?!?*??X??){?G?zd?1?*??X??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6J?!??T??)?)-C??6J?1??T??)?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q?`8ب??"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JDESKTOP-D9SGMFS: Failed to load libcupti (is it installed and accessible?)
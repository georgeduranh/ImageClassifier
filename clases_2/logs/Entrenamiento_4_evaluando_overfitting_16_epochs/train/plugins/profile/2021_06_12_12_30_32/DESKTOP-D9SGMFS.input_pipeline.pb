	]?C???L@]?C???L@!]?C???L@	6??"?'??6??"?'??!6??"?'??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:]?C???L@?	???A?z6?fL@Y??o_??rEagerKernelExecute 0*	    8?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????9#??!I??īHR@)??0?*??1L??d?KP@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapԚ?????!????1@)EGr????1Цm(@:Preprocessing2U
Iterator::Model::ParallelMapV2?N@aã?!???>.?@)?N@aã?1???>.?@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?ܵ?|У?!?K??d?@)z6?>W[??1???,@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?U???؟?!hӀ6@)?U???؟?1hӀ6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate/n????!+?"?*@)X?5?;N??1ra?q@:Preprocessing2F
Iterator::Model??HP??!?īH??@)?o_???1=??=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate?{??Pk??!???ۡ??)??ZӼ???1??P???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!??!a?qa??)??ׁsF??1q?GOp??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?%䃞???!?GOp?R@)Ǻ???v?1?qa??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?g??s?u?!?U$^E???)?g??s?u?1?U$^E???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?s?!?=????)a2U0*?s?1?=????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor_?Q?[?!=????)_?Q?[?1=????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice????MbP?!?B!???)????MbP?1?B!???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensorǺ???F?!?qa??)Ǻ???F?1?qa??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no95??"?'??I??n,??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?	????	???!?	???      ??!       "      ??!       *      ??!       2	?z6?fL@?z6?fL@!?z6?fL@:      ??!       B      ??!       J	??o_????o_??!??o_??R      ??!       Z	??o_????o_??!??o_??b      ??!       JCPU_ONLYY5??"?'??b q??n,??X@
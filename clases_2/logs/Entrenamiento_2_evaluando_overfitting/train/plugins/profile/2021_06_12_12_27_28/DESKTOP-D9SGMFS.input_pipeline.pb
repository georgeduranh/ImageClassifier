	?z?G?J@?z?G?J@!?z?G?J@	 ??P?	?? ??P?	??! ??P?	??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?z?G?J@??Ƽ?A4??7?bJ@Y?U???ؿ?rEagerKernelExecute 0*	effff?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapd?]K???!??z???L@)?ZB>????1??#??#H@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map3ı.n???!?Ю?Ю=@)???߾??1?8?87@:Preprocessing2U
Iterator::Model::ParallelMapV2???Mb??!?/??/?@)???Mb??1?/??/?@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat46<?R??!?=??=?@)?ܵ?|У?1c??b??@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::PrefetchǺ????!??????@)Ǻ????1??????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate$????ۗ?!?|??|?@)??ͪ?Ֆ?1??o??o
@:Preprocessing2F
Iterator::Model7?[ A??!????#@)??ZӼ???1xX/xX/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate'???????!??^??^??)?? ?rh??1dt'dt'??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ?????!W?W???)y?&1?|?1p??p????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangen??t?!7?<7?<??)n??t?17?<7?<??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??(???!9?9*N@)a2U0*?s?1&?&???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4q?!T??S????)?J?4q?1T??S????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensorǺ???V?!W?W???)Ǻ???V?1W?W???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor????MbP?!??????)????MbP?1??????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice-C??6J?!?Y?Y??)-C??6J?1?Y?Y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9 ??P?	??I?W??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Ƽ???Ƽ?!??Ƽ?      ??!       "      ??!       *      ??!       2	4??7?bJ@4??7?bJ@!4??7?bJ@:      ??!       B      ??!       J	?U???ؿ??U???ؿ?!?U???ؿ?R      ??!       Z	?U???ؿ??U???ؿ?!?U???ؿ?b      ??!       JCPU_ONLYY ??P?	??b q?W??X@
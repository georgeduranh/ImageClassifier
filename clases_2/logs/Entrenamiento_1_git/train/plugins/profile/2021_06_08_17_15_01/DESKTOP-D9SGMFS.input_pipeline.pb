	?K7?A?Q@?K7?A?Q@!?K7?A?Q@	(??&??(??&??!(??&??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?K7?A?Q@1?*????A'1??Q@Y?L?J???rEagerKernelExecute 0*	23333?w@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???H??!???+??H@)8gDio???1w&{??dF@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map2??%䃾?!4c7,?%?@)??JY?8??1?!N?R?6@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??ǘ????!??ҷ?? @)??e?c]??1???0??@:Preprocessing2F
Iterator::Model???JY???!ϳ??\)@)??+e???1?Ļf@:Preprocessing2U
Iterator::Model::ParallelMapV2??0?*??!??E??@)??0?*??1??E??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?j+??ݓ?!ր?F@)???߾??1?4?|?@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch/?$???!w????@)/?$???1w????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?@??ǘ??!r!?D+L@)9??v??z?1?`H??,??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice-C??6z?!??r5????)-C??6z?1??r5????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?+e?Xw?!?;????)?+e?Xw?1?;????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeU???N@s?!?&@?\???)U???N@s?1?&@?\???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?C?Οi??)Ǻ???V?1?C?Οi??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9)??&??I??O??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1?*????1?*????!1?*????      ??!       "      ??!       *      ??!       2	'1??Q@'1??Q@!'1??Q@:      ??!       B      ??!       J	?L?J????L?J???!?L?J???R      ??!       Z	?L?J????L?J???!?L?J???b      ??!       JCPU_ONLYY)??&??b q??O??X@
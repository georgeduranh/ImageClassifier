
€
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*28‘έ@‘έH‘έXb;gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropInputhu  ΘA
~
;volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1~*28₯θ³@₯θ³H₯θ³bsequential_2/Conv2/Reluhu  ΘA
{
volta_sgemm_64x64_nt~@*@2$8Θϋ@ΘϋHΘϋXb<gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropFilterhu  HB
{
volta_sgemm_64x64_nt~@*@2$8Μ@ΜHΜXb<gradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilterhu  HB
ι
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2@8ςΟ€@ςΟ€HςΟ€Xb<gradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilterhu  B
ι
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2@8@HXb<gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropFilterhu  B
η
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2@8σΊ@σΊHσΊXb<gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropFilterhu  HB
ς
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*2(8υ―ν@υ―νHυ―νb)gradient_tape/sequential_2/Conv1/ReluGradhu  ΘB
{
volta_sgemm_64x64_nt~@*@2$8υμ@υμHυμXb<gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropFilterhu  HB
}
volta_sgemm_128x64_ntz`*2$8ΥΛα@ΥΛαHΥΛαXb;gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropInputhu  HB
ς
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*2(8Φςά@ΦςάHΦςάb)gradient_tape/sequential_2/Conv2/ReluGradhu  ΘB

Ήvoid cudnn::pooling_bw_kernel_max_nchw_fully_packed_large<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor, int)#  π*2@8Π@ΠHΠb7gradient_tape/sequential_2/Pooling1/MaxPool/MaxPoolGradhu  HB
}
volta_cgemm_64x32_tn^h*2 8ζΑ@ζΑHζΑXb<gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropFilterhu  HB
V
volta_cgemm_64x32_tn^h*2 8Ψ«€@Ψ«€HΨ«€bsequential_2/Conv4/Reluhu  HB
W
volta_sgemm_128x64_nnzb*2$8ω?’@ω?’Hω?’bsequential_2/Conv3/Reluhu  HB

void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ΐ*28ΨΗ‘@ά¨OHόRXb<gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropFilterhu  HB
ι
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)@*2 8₯@₯H₯Xb<gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropFilterhu  B
ι
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)±*2@8ϊ©@ϊ©Hϊ©Xb;gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropInputhu  HB
ι
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)±*2@8Ϊ@ΪHΪXb;gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropInputhu  HB
Δ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)P*2@8ϊΩ@ϊΩHϊΩbsequential_2/Conv3/Reluhu  B
κ
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)P*2@8ϊ~@ϊ~Hϊ~Xb;gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropInputhu  B
z
volta_sgemm_128x64_ntz`*2$8ϋΟr@ϋΟrHϋΟrXb;gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropInputhu  HB
ο
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*2(8ϋοj@ϋοjHϋοjb)gradient_tape/sequential_2/Conv3/ReluGradhu  ΘB
ο
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*2(8ό]@ό]Hό]b)gradient_tape/sequential_2/Conv4/ReluGradhu  ΘB
e
%volta_scudnn_128x32_relu_medium_nn_v1**2@8κ[@κ[Hκ[bsequential_2/Conv1/Reluhu  HB
«
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28ά’X@ά’XHά’XXb<gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropFilterhu?A€B
Χ
void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)9  *ς28ά’S@ά’SHά’Sbsequential_2/Pooling1/MaxPoolhuω½B

Κvoid fft2d_c2r_32x32<float, false, true, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)@ ΐ*28όΏR@όΏRHόΏRbsequential_2/Conv4/Reluhu  HB
ΰ
void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ΐ*28ό΄Q@ό΄QHό΄Qbsequential_2/Conv4/Reluhu  HB

΄void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) Θ4*28©K@©KH©Kb7gradient_tape/sequential_2/Pooling2/MaxPool/MaxPoolGradhu  ΘB
Ώ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)±*2@@8έB@έBHέBbsequential_2/Conv3/Reluhu  HB
δ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*28½?@½?H½?Xb<gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropFilterhu  HB
ι
void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)P*2@@8έξ>@έξ>Hέξ>Xb;gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropInputhu  B
θ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)70*28ύΣ;@ύΣ;HύΣ;b4gradient_tape/sequential_2/Conv1/BiasAdd/BiasAddGradhu  ΘB
θ
void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)70*28ύΦ9@ύΦ9HύΦ9b4gradient_tape/sequential_2/Conv2/BiasAdd/BiasAddGradhu  ΘB
ώ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*2(8ύΏ3@ύΏ3HύΏ3b$Adam/Adam/update_8/ResourceApplyAdamhu  ΘB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ύέ.@ύέ.Hύέ.b$sequential_2/dropout_6/dropout/Mul_1hu  ΘB
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28Ύ³-@Ύ³-HΎ³-b2gradient_tape/sequential_2/dropout_6/dropout/Mul_1hu  ΘB
Χ
void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)9 ΐ*Δ2 8ώο#@ώο#Hώο#bsequential_2/Pooling2/MaxPoolhu  B
j
volta_sgemm_128x128_tnv*2Δ8ώ―#@ώ―#Hώ―#Xb)gradient_tape/sequential_2/dense_4/MatMulhu  HB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28?― @?― H?― b0gradient_tape/sequential_2/dropout_6/dropout/Mulhu  ΘB
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ί @ί Hί b"sequential_2/dropout_6/dropout/Mulhu  ΘB
ύ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)(*2@8²@²H²b4gradient_tape/sequential_2/Conv3/BiasAdd/BiasAddGradhu  ΘB
Ω
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2(8Ύ?@Ύ?HΎ?b_sequential_2/dropout_6/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ΘB
Y
volta_sgemm_128x64_nnzb*2
8Ο@ΟHΟXbsequential_2/dense_4/MatMulhu  HB
h
volta_sgemm_128x64_ntz`*28ΏΜ@ΏΜHΏΜb+gradient_tape/sequential_2/dense_4/MatMul_1hu  HB
δ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)@Α*2 8?@?H?Xb<gradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilterhu  HB
³
Λvoid fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)? ΐ*28?«@?«H?«Xb<gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropFilterhu  HB
ύ
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)(*2@8?Α@?ΑH?Αb4gradient_tape/sequential_2/Conv4/BiasAdd/BiasAddGradhu  ΘB
ΰ
void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ΐ*28?ί@?ίH?ίbsequential_2/Conv4/Reluhu  HB
x
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28?ύ@?ύH?ύb+sequential_2/dropout_6/dropout/GreaterEqualhu  ΘB
w
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2 8??@??H??b2gradient_tape/sequential_2/dropout_7/dropout/Mul_1hu  ΘB
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2 8ί@ίHίb$sequential_2/dropout_7/dropout/Mul_1hu  ΘB
υ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)'*2(8ΏΠ@ΏΠHΏΠb;sequential_2/dropout_6/dropout/random_uniform/RandomUniformhu  ΘB
?
¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)@!*288ΰ΅@ΰ΅Hΰ΅b]gradient_tape/sequential_2/Pooling2/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB

¨void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)@!*288ΐΉ@ΐΉHΐΉbLsequential_2/dropout_7/dropout/Mul_1-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ΘB
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2 8ΰσ@ΰσHΰσb0gradient_tape/sequential_2/dropout_7/dropout/Mulhu  ΘB
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*2 8ΐρ@ΐρHΐρb"sequential_2/dropout_7/dropout/Mulhu  ΘB
ζ
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*): *2|8 γ@ γH γbgradient_tape/sequential_2/dropout_6/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul-1-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
Ω
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2(8?²@?²H?²b_sequential_2/dropout_7/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ΘB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2(8ΐ@ΐHΐb#sequential_2/dropout_7/dropout/Casthu  ΘB
x
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*2 8Ώω@ΏωHΏωb+sequential_2/dropout_7/dropout/GreaterEqualhu  ΘB
΅
ͺvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)@`*28 ψ@ ψH ψbbgradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB
υ
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)'*2(8Λ@ΛHΛb;sequential_2/dropout_7/dropout/random_uniform/RandomUniformhu  ΘB
Ϊ
«void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 256, 32, 32, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*): *288?Ώ@?ΏH?Ώbsequential_2/dropout_7/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ΘB

Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28ΰΔ@ΰΔHΰΔbsequential_2/Conv4/Reluhu  aA
ώ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*2(8ΰ£@ΰ£Hΰ£b$Adam/Adam/update_6/ResourceApplyAdamhu  ΘB
λ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)>H* 28ΰ@ΰHΰXb<gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropFilterhu  ΘB
π
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)=*28@Hb:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ΘB
η
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)@H* 28@HXb;gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropInputhu  ΘB
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)>H* 28 s@ sH sXb<gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropFilterhu  ΘB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2(8ΰj@ΰjHΰjXb;gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropInputhu  ΘB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2(8i@iHiXb<gradient_tape/sequential_2/Conv4/Conv2D/Conv2DBackpropFilterhu  ΘB
θ
void cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)>H* 28 g@ gH gXb<gradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilterhu  ΘB
λ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)@*28`@`H`b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*2(8`@`H`b$Adam/Adam/update_4/ResourceApplyAdamhu  ΘB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2(8P@PHPbsequential_2/Conv4/Reluhu  ΘB
`
volta_sgemm_32x32_sliced1x4_nnV*28P@PHPXbsequential_2/dense_5/MatMulhu  ΘA
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2(8ΐO@ΐOHΐOXb;gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropInputhu  ΘB
ε
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long):*28ΰ@@ΰ@Hΰ@b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ΘB
Ύ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)@H* 28@@@H@bsequential_2/Conv3/Reluhu  ΘB
δ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8@@@H@Xb;gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropInputhu  ΘB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2(8@@@H@bsequential_2/Conv3/Reluhu  ΘB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2(8@@@H@Xb<gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropFilterhu  ΘB
δ
void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)@H* 28??@??H??Xb;gradient_tape/sequential_2/Conv3/Conv2D/Conv2DBackpropInputhu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*2$8 ;@ ;H ;b$Adam/Adam/update_2/ResourceApplyAdamhu  ΘB
n
volta_sgemm_32x32_sliced1x4_ntV*28ΐ0@ΐ0Hΐ0b+gradient_tape/sequential_2/dense_5/MatMul_1hu  HB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$80@0H0Xb<gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropFilterhu  ΘB
e
volta_sgemm_128x32_tn7*28 .@ .H .Xb)gradient_tape/sequential_2/dense_5/MatMulhu  ΘB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*2$8ΰ-@ΰ-Hΰ-bsequential_2/Conv2/Reluhu  ΘB
ω
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28 ,@ ,H ,b"Adam/Adam/update/ResourceApplyAdamhu  ΘB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*2(8ΐ+@ΐ+Hΐ+bsequential_2/dense_4/BiasAddhu  ΘB
ς
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)'*2(8 +@ +H +b;sequential_2/dropout_8/dropout/random_uniform/RandomUniformhu  ΘB
λ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  2 8ΐ%@ΐ%Hΐ%b4gradient_tape/sequential_2/Conv3/BiasAdd/BiasAddGradhu  ΘB
Ε
ιvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)**28 $@ $H $b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ΘB
ώ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28#@#H#b4gradient_tape/sequential_2/Conv3/BiasAdd/BiasAddGradhu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ΐ!@ΐ!Hΐ!b$Adam/Adam/update_7/ResourceApplyAdamhu  ΘB
­
Ρvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)$*28!@!H!b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ΘB
ή
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)4D* 28  @  H  Xb;gradient_tape/sequential_2/Conv2/Conv2D/Conv2DBackpropInputhu  ―B
ώ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28  @  H  b4gradient_tape/sequential_2/Conv1/BiasAdd/BiasAddGradhu  ΘB
t
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28 @ H b+sequential_2/dropout_8/dropout/GreaterEqualhu  ΘB
z
Bcask_cudnn::computeOffsetsKernel(cask_cudnn::ComputeOffsetsParams)&*28 @ H bsequential_2/Conv1/Reluhu  ΘB
Ϊ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*2(8 @ H bsequential_2/dense_4/Reluhu  ΘB

γvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bAssignAddVariableOp_3hu  ΘB
?	
£	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)(*28 @ H b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ΘB
¬
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *2(8 @ H b6gradient_tape/sequential_2/dense_4/BiasAdd/BiasAddGradhu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28 @ H b$Adam/Adam/update_1/ResourceApplyAdamhu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28 @ H b$Adam/Adam/update_3/ResourceApplyAdamhu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28 @ H b$Adam/Adam/update_5/ResourceApplyAdamhu  ΘB
ϋ
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28 @ H b$Adam/Adam/update_9/ResourceApplyAdamhu  ΘB
λ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28 @ H b4gradient_tape/sequential_2/Conv1/BiasAdd/BiasAddGradhu  ΘB
λ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28 @ H b4gradient_tape/sequential_2/Conv2/BiasAdd/BiasAddGradhu  ΘB
λ
void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  2 8 @ H b4gradient_tape/sequential_2/Conv4/BiasAdd/BiasAddGradhu  ΘB
ώ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b4gradient_tape/sequential_2/Conv2/BiasAdd/BiasAddGradhu  ΘB
ώ
¦void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!*  28 @ H b4gradient_tape/sequential_2/Conv4/BiasAdd/BiasAddGradhu  ΘB
Ύ
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28 @ H bsequential_2/Conv1/Reluhu  ΘB
ε
void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28 @ H Xb<gradient_tape/sequential_2/Conv1/Conv2D/Conv2DBackpropFilterhu  ΘB

²void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)!* 28@Hb6gradient_tape/sequential_2/dense_5/BiasAdd/BiasAddGradhu ’B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*28ΰ@ΰHΰb
LogicalAndhu  ΘB
ΰ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28@Hb;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1hu  ΘB
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*28ΐ@ΐHΐbAdam/addhu  ΘB
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*28 @ H bEqualhu  ΘB
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H b0gradient_tape/sequential_2/dropout_8/dropout/Mulhu  ΘB
΅
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *28@HbArgMax_1hu  ΘB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ΐ@ΐHΐb%Adam/Adam/update_11/ResourceApplyAdamhu  ΘB
ϋ
Σvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bCast_2hu  ΘB

Ωvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@HbAssignAddVariableOp_4hu  ΘB
³
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *28ΰ@ΰHΰbArgMaxhu  ΘB
e
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28ΐ@ΐHΐb$sequential_2/dropout_8/dropout/Mul_1hu  ΘB
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H b2gradient_tape/sequential_2/dropout_8/dropout/Mul_1hu  ΘB
ξ
‘void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*2(8ΐ@ΐHΐb+gradient_tape/sequential_2/dense_4/ReluGradhu  ΘB
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28@HbAdam/Powhu  ΘB
¬
Εvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΐ@ΐHΐbEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhu  ΘB

Ωvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΐ@ΐHΐbAdam/Adam/AssignAddVariableOphu  ΘB
ό
΅void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ΰ@ΰHΰb%Adam/Adam/update_10/ResourceApplyAdamhu  ΘB

Εvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@Hb,categorical_crossentropy/weighted_loss/valuehu  ΘB

Εvoid tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*2@8@Hbsequential_2/dense_5/Softmaxhu  ΘB
κ
void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*)-*28ΰ@ΰHΰb+gradient_tape/sequential_2/dense_5/MatMul_1hu  ΘB
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28ΐ@ΐHΐb
Adam/Pow_1hu  ΘB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2(8ΐ@ΐHΐb#sequential_2/dropout_8/dropout/Casthu  ΘB
α
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**2@8ΐ@ΐHΐbsequential_2/dense_5/Softmaxhu  ΘB
Ο
void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool)%*28 @ H bsequential_2/dense_5/Softmaxhu  ΘB

γvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@HbAssignAddVariableOphu  ΘB
ω
Σvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bCasthu  ΘB
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Hb"sequential_2/dropout_8/dropout/Mulhu  ΘB
Έ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)4D* 28@Hbsequential_2/Conv2/Reluhu  ―B
σ
Εvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΰ@ΰHΰbdiv_no_nan_1hu  ΘB
­
Σvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΰ@ΰHΰb8categorical_crossentropy/weighted_loss/num_elements/Casthu  ΘB
χ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*28@Hb:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ΘB
ύ
Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΰ@ΰHΰbCast_1hu  ΘB

Υvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΰ@ΰHΰbAdam/Cast_1hu  ΘB

γvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ΰ@ΰHΰbAssignAddVariableOp_1hu  ΘB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28 @ H bsequential_2/dense_5/BiasAddhu  ΘB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@HbLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhu  ΘB
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@HbMulhu  ΘB
ρ
Εvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@Hb
div_no_nanhu  ΘB

γvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@HbAssignAddVariableOp_2hu  ΘB
λ
Βvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28@HbSum_2hu  ΘB

Βvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28@Hb*categorical_crossentropy/weighted_loss/Sumhu  ΘB
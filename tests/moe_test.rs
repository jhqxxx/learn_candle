use learn_candle::moe::SparseMoeBlock;
use candle_nn::{VarMap, VarBuilder};
use candle_core::{Device, Tensor};

#[test]
fn test_moe() {
    let device = Device::cuda_if_available(0).unwrap();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let embedding_dim = 32;
    let hidden_dim = 128;
    let num_experts = 8;

    let moe = SparseMoeBlock::new(vb, num_experts, embedding_dim, hidden_dim, true, 2).unwrap();
    let input = Tensor::rand(0.0f32, 1.0, (2, 2, embedding_dim), &device).unwrap();
    let out = moe.forward(&input).unwrap();
}
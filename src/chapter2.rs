use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, Optimizer, SGD, VarBuilder, VarMap, linear, loss, ops};

use crate::utils::dataset::{DataLoader, DemoDataset};
use crate::utils::net::print_varmap;
use rand::seq::SliceRandom;

pub struct SimpleModel {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
}

impl SimpleModel {
    pub fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let linear1 = linear(in_dim, hidden_dim, vb.pp("linear1"))?;
        let linear2 = linear(hidden_dim, hidden_dim, vb.pp("linear2"))?;
        let linear3 = linear(hidden_dim, out_dim, vb.pp("linear3"))?;
        Ok(Self {
            linear1,
            linear2,
            linear3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.silu()?;
        let x = self.linear2.forward(&x)?;
        let x = x.silu()?;
        let x = self.linear3.forward(&x)?;
        Ok(x)
    }
}

pub fn train() -> Result<()> {
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = SimpleModel::new(vb, 2, 20, 2)?;
    let _ = print_varmap(&varmap)?;
    let x_train = Tensor::from_vec(
        vec![-1.2f32, 3.1, -0.9, 2.9, -0.5, 2.6, 2.3, -1.1, 2.7, -1.5],
        (5, 2),
        &device,
    )?;
    let y_train = Tensor::from_vec(vec![0u32, 0, 0, 1, 1], 5, &device)?;
    let x_val = Tensor::from_vec(vec![-0.8f32, 2.8, 2.6, -1.6], (2, 2), &device)?;
    let y_val = Tensor::from_vec(vec![0u32, 1], (2,), &device)?;
    let train_dataset = DemoDataset::new(x_train, y_train)?;
    let val_dataset = DemoDataset::new(x_val, y_val)?;
    let mut train_loader = DataLoader::new(train_dataset, 5, true)?;
    let mut val_loader = DataLoader::new(val_dataset, 2, false)?;

    // 训练
    // let mut sgd = SGD::new(varmap.all_vars(), 0.01)?;
    // let epochs = 3;
    // for epoch in 0..epochs {
    //     let _ = train_loader.reset()?;
    //     let _ = val_loader.reset()?;
    //     for batch in &mut train_loader {
    //         let (x_, y_) = batch?;
    //         let predict = model.forward(&x_)?;
    //         let loss_ = loss::cross_entropy(&predict, &y_)?;
    //         sgd.backward_step(&loss_)?;
    //         println!("epoch: {} train loss: {}", epoch, loss_);
    //     }
    //     for batch in &mut val_loader {
    //         let (x_, y_) = batch?;
    //         let predict = model.forward(&x_)?;
    //         let loss_ = loss::cross_entropy(&predict, &y_)?;
    //         println!("epoch: {} val loss: {}", epoch, loss_);
    //     }
    // }

    // 推理
    let _ = train_loader.reset()?;
    let _ = val_loader.reset()?;
    for batch in &mut train_loader {
        let (x_, y_) = batch?;
        let predict = model.forward(&x_)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("init label: {}", label);
        println!("train true label: {}", y_);
        // label == y_ 1/0  1 / 2 =1/0.5
        println!(
            "train acc: {}",
            label
                .eq(&y_)?
                .sum(0)?
                .to_dtype(candle_core::DType::F32)?
                .affine(1.0 / (x_.dim(0)? as f64), 0.0)?
        );
    }
    for batch in &mut val_loader {
        let (x_, y_) = batch?;
        let predict = model.forward(&x_)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("val label: {}", label);
        println!("val true label: {}", y_);
        println!(
            "val acc: {}",
            label
                .eq(&y_)?
                .sum(0)?
                .to_dtype(candle_core::DType::F32)?
                .affine(1.0 / (x_.dim(0)? as f64), 0.0)?
        );
    }

    // 模型保存
    // varmap.save("model.safetensors")?;
    // 模型加载
    varmap.load("model.safetensors")?;
    let _ = train_loader.reset()?;
    let _ = val_loader.reset()?;
    for batch in &mut train_loader {
        let (x_, y_) = batch?;
        let predict = model.forward(&x_)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("train label: {}", label);
        println!("train true label: {}", y_);
        // label == y_ 1/0  1 / 2 =1/0.5
        println!(
            "train acc: {}",
            label
                .eq(&y_)?
                .sum(0)?
                .to_dtype(candle_core::DType::F32)?
                .affine(1.0 / (x_.dim(0)? as f64), 0.0)?
        );
    }
    for batch in &mut val_loader {
        let (x_, y_) = batch?;
        let predict = model.forward(&x_)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("val label: {}", label);
        println!("val true label: {}", y_);
        println!(
            "val acc: {}",
            label
                .eq(&y_)?
                .sum(0)?
                .to_dtype(candle_core::DType::F32)?
                .affine(1.0 / (x_.dim(0)? as f64), 0.0)?
        );
    }
    Ok(())
}

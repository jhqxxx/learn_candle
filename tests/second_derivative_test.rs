use candle_core::{Device, Var, Result};

#[test]
fn test_second_derivative() -> Result<()> {
    //  cargo test test_second_derivative -- --nocapture
    let device = Device::cuda_if_available(0)?;
    // y = x^3
    // 一阶导: 3x^2,当x = 3时,结果为: 27
    // 二阶导: 3*2*x,当x=3时,结果为: 18
    let var1 = Var::from_vec(vec![3f32], 1, &device)?;
    let y = var1.powf(3f64)?;
    let grad = y.backward()?;
    let var1_grad = grad.get(&var1).unwrap();
    println!("var1 grad: {}", var1_grad);
    let grad_2 = var1_grad.backward()?;
    let var1_grad_2 = grad_2.get(&var1).unwrap();
    println!("var1 grad2: {}", var1_grad_2);
    Ok(())
}
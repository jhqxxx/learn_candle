use candle_core::{Error, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, VarMap, conv2d};
use plotters::prelude::*;

#[allow(unused)]
pub fn print_varmap(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut param_count = 0;
    for (key, value) in data.iter() {
        println!("{key}");
        println!("{:?}", value);
        param_count += value.elem_count();
    }
    println!("{param_count}");
    Ok(())
}

#[allow(unused)]
pub fn plot_loss_curves(
    train_losses: &[f32],
    val_losses: &[f32],
    filename: &str,
    title: &str,
) -> Result<()> {
    let root_area = BitMapBackend::new(filename, (1200, 800)).into_drawing_area();
    root_area
        .fill(&WHITE)
        .map_err(|e| Error::Msg(format!("bitmap fill white color wrong")))?;
    let max_y = train_losses
        .iter()
        .chain(val_losses.iter())
        .max_by(|a, b| a.total_cmp(b))
        .copied()
        .unwrap_or(5.0)
        * 1.1;
    let mut chart = ChartBuilder::on(&root_area)
        .caption(title, ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..train_losses.len() as i32, 0.0..max_y)
        .map_err(|e| Error::Msg(format!("chart create wrong")))?;

    chart
        .configure_mesh()
        .draw()
        .map_err(|e| Error::Msg(format!("chart configure_mesh wrong")))?;

    // 训练 loss 曲线（蓝色）
    chart
        .draw_series(LineSeries::new(
            train_losses.iter().enumerate().map(|(i, &v)| (i as i32, v)),
            BLUE,
        ))
        .map_err(|e| Error::Msg(format!("chart draw_series wrong")))?
        .label("Train Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // 验证 loss 曲线（红色）
    chart
        .draw_series(LineSeries::new(
            val_losses.iter().enumerate().map(|(i, &v)| (i as i32, v)),
            RED,
        ))
        .map_err(|e| Error::Msg(format!("chart draw_series wrong")))?
        .label("Val Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| Error::Msg(format!("chart configure_series_labels wrong")))?;

    root_area
        .present()
        .map_err(|e| Error::Msg(format!("root_area.present wrong")))?;
    Ok(())
}

#[allow(unused)]
pub struct ResDWConv2d {
    dw_conv: Conv2d,
}
impl ResDWConv2d {
    #[allow(unused)]
    pub fn new(vb: VarBuilder, channels: usize, ksize: usize) -> Result<Self> {
        let d_cfg = Conv2dConfig {
            padding: ksize / 2,
            stride: 1,
            groups: channels,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let dw_conv = conv2d(channels, channels, ksize, d_cfg, vb.pp("dw_conv"))?;

        Ok(Self { dw_conv })
    }

    #[allow(unused)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_ = self.dw_conv.forward(x)?;
        let x = x_.add(x)?;
        Ok(x)
    }
}

pub trait CustomModule {
    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor>;
}

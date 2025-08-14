use candle_core::{Device, Error, Result, Tensor};

mod chapter3;
mod chapter4;
mod utils;
mod moe;
use moe::SparseMoeBlock;
use candle_nn::{VarMap, VarBuilder};

use chapter4::train_img_main;



fn main() -> Result<()> {
    train_img_main()?;
    Ok(())
}

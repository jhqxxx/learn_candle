use candle_core::Result;

mod chapter3;
mod chapter4;
mod utils;
use chapter4::train_img_main;

fn main() -> Result<()> {
    let _ = train_img_main()?;
    Ok(())
}

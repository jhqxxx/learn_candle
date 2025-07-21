use candle_core::{DType, Device, Result, Tensor};
use rand::prelude::SliceRandom;

#[allow(unused)]
pub trait Dataset {
    fn len(&self) -> Result<usize>;
    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)>;
    fn shuffle(&mut self) -> Result<()>;
}

#[macro_export]
macro_rules! impl_dataset {
    ($t:ty) => {
        impl Dataset for $t {
            fn len(&self) -> Result<usize> {
                Ok(self.input_ids.dim(0)?)
            }
            fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
                let inputs = self.input_ids.narrow(0, start, end - start)?;
                let targets = self.target_ids.narrow(0, start, end - start)?;
                Ok((inputs, targets))
            }
            fn shuffle(&mut self) -> Result<()> {
                let len = self.len()?;
                let mut idx: Vec<u32> = (0..len).map(|i| i as u32).collect();
                let mut rng = rand::rng();
                idx.shuffle(&mut rng);
                let idx_tensor = Tensor::from_vec(idx, (len,), self.input_ids.device())?;
                self.input_ids = self.input_ids.index_select(&idx_tensor, 0)?;
                self.target_ids = self.target_ids.index_select(&idx_tensor, 0)?;
                Ok(())
            }
        }
    };
}

pub struct DemoDataset {
    input_ids: Tensor,
    target_ids: Tensor,
}

impl DemoDataset {
    #[allow(unused)]
    pub fn new(x: Tensor, y: Tensor) -> Result<Self> {
        Ok(Self {
            input_ids: x,
            target_ids: y,
        })
    }

    #[allow(unused)]
    pub fn get_idx(&self, idx: usize) -> Result<(Tensor, Tensor)> {
        let x_idx = self.input_ids.narrow(0, idx, 1)?;
        let y_idx = self.target_ids.narrow(0, idx, 1)?;
        Ok((x_idx, y_idx))
    }
}

impl_dataset!(DemoDataset);

#[derive(Clone)]
pub struct ImgDataset {
    img_vec: Vec<Vec<u8>>,
    label_vec: Vec<u8>,
    device: Device,
    mean: Tensor,
    std: Tensor,
}
impl ImgDataset {
    pub fn new(img_vec: Vec<Vec<u8>>, label_vec: Vec<u8>, device: &Device) -> Result<Self> {
        let mean = Tensor::new(&[0.5f32, 0.5, 0.5], &device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&[0.5f32, 0.5, 0.5], &device)?.reshape((3, 1, 1))?;
        Ok(Self {
            img_vec,
            label_vec,
            device: device.clone(),
            mean,
            std,
        })
    }
}
impl Dataset for ImgDataset {
    fn len(&self) -> Result<usize> {
        Ok(self.img_vec.len())
    }
    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
        // （bs, h, w, channel)-> (bs, c, h, w)
        let img_tensor = Tensor::new(self.img_vec[start..end].to_vec(), &self.device)?
            .reshape((end - start, 224, 224, 3))?
            .permute((0, 3, 1, 2))?;
        // [0, 255] -> [0, 1]
        let img_tensor = img_tensor.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)?;
        // Normalize: (x - mean) / std
        let img_tensor = img_tensor
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)?;
        let label_tensor = Tensor::from_vec(
            self.label_vec[start..end].to_vec(),
            end - start,
            &self.device,
        )?;
        Ok((img_tensor, label_tensor))
    }

    fn shuffle(&mut self) -> Result<()> {
        let mut rng = rand::rng();
        let mut indices: Vec<(usize, &Vec<u8>)> = self.img_vec.iter().enumerate().collect();
        indices.shuffle(&mut rng);

        // Reorder both vectors based on shuffled indices
        let (new_img_vec, new_label_vec): (Vec<_>, Vec<_>) = indices
            .into_iter()
            .map(|(i, img)| (img.clone(), self.label_vec[i]))
            .unzip();

        self.img_vec = new_img_vec;
        self.label_vec = new_label_vec;
        Ok(())
    }
}
#[allow(unused)]
pub struct DataLoader<'a> {
    dataset: Box<dyn Dataset + 'a>,
    batch_size: usize,
    current_index: usize,
    shuffle: bool,
}

#[allow(unused)]
impl<'a> DataLoader<'a> {
    pub fn new<D: Dataset + 'a>(dataset: D, batch_size: usize, shuffle: bool) -> Result<Self> {
        Ok(Self {
            dataset: Box::new(dataset),
            batch_size,
            current_index: 0,
            shuffle,
        })
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.dataset.len()?)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.current_index = 0;
        if self.shuffle {
            let _ = self.dataset.shuffle()?;
        }
        Ok(())
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Result<(Tensor, Tensor)>;
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_index * self.batch_size;
        let end = std::cmp::min(start + self.batch_size, self.dataset.len().ok()?);
        if start >= end {
            return None;
        }
        let batch = self.dataset.get_batch(start, end).ok()?;
        self.current_index += 1;
        Some(Ok(batch))
    }
}
